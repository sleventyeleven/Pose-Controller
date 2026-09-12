from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from pose_controller.inference.pose import PoseResult
from pose_controller.tracking.kalman import BoxKalmanFilter
from pose_controller.tracking.reid import ReidEmbedder, cosine_similarity

BoxXYXY = tuple[float, float, float, float]


def _iou(box_a: BoxXYXY, box_b: BoxXYXY) -> float:
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    kalman: BoxKalmanFilter
    time_since_update: int = 0
    hits: int = 1
    predicted_box: BoxXYXY = field(default=(0.0, 0.0, 0.0, 0.0))
    last_embedding: np.ndarray | None = None


@dataclass
class _LostTrack:
    track_id: int
    embedding: np.ndarray
    frames_lost: int = 0


class Tracker:
    """SORT-style multi-object tracker: a constant-velocity Kalman filter
    per track, matched to each frame's detections by IoU via the Hungarian
    algorithm (`scipy.optimize.linear_sum_assignment`).

    Gives each detected person a stable `track_id` across frames --
    including through brief occlusion, since unmatched tracks are kept
    alive (coasting on their Kalman prediction) for `max_age` frames
    before being dropped, rather than being deleted the instant a
    detection is missed for one frame.

    Optionally also re-identifies someone who fully left frame and
    returned, past `max_age`: when a `ReidEmbedder` is supplied, a track
    that ages out is kept a while longer in a "lost gallery" (keyed by its
    last-known appearance embedding, not position) for `reid_gallery_ttl`
    frames. A new, otherwise-unmatched detection is checked against that
    gallery by cosine similarity before allocating a fresh ID -- a
    confident match revives the old `track_id` instead. Without an
    embedder, the tracker still works, just without that revival (a new
    ID is always allocated for a detection that doesn't IoU-match an
    active track).

    Usage: call `update()` once per frame with that frame's detections
    (order-independent, no `track_id` needed on input) and the frame
    itself (only needed for re-ID cropping -- omit if no embedder is
    configured); returns the same detections with `track_id` populated.
    Detections with no computable bbox (fewer than 2 sufficiently-visible
    keypoints -- see `pose.bbox_from_keypoints`) are dropped rather than
    tracked, since they're too unreliable to act on anyway.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 1,
        embedder: ReidEmbedder | None = None,
        reid_similarity_threshold: float = 0.6,
        reid_gallery_ttl: int = 300,
    ):
        """
        iou_threshold: minimum IoU between a track's predicted box and a
            detection's box to count as the same person.
        max_age: frames a track survives with no matching detection before
            being dropped (at ~15-30fps, 30 frames is roughly 1-2 seconds).
        min_hits: consecutive matched frames a track needs before its
            `track_id` is reported (filters out one-off false-positive
            detections from ever getting an ID at all). Default 1 reports
            immediately.
        embedder: optional `ReidEmbedder` for appearance-based re-ID. If
            None, tracks that age out are gone for good (position/motion
            tracking only).
        reid_similarity_threshold: minimum cosine similarity (embeddings
            are L2-normalized, so this is a dot product in [-1, 1]) between
            a new detection and a lost track's last embedding to revive
            that track's ID. Not empirically tuned against a benchmark --
            a reasonable starting point (see docs/backlog.md).
        reid_gallery_ttl: additional frames (beyond `max_age`) a track's
            embedding is kept in the "lost gallery" for possible revival
            before being forgotten entirely. Default 300 at ~15-30fps is
            roughly 10-20 seconds.
        """
        self._tracks: list[_Track] = []
        self._lost_gallery: list[_LostTrack] = []
        self._next_id = 1
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._min_hits = min_hits
        self._embedder = embedder
        self._reid_similarity_threshold = reid_similarity_threshold
        self._reid_gallery_ttl = reid_gallery_ttl

    def update(
        self, detections: list[PoseResult], frame_bgr: np.ndarray | None = None
    ) -> list[PoseResult]:
        trackable = [d for d in detections if d.bbox is not None]

        for track in self._tracks:
            track.predicted_box = track.kalman.predict()

        matches, unmatched_track_idx, unmatched_det_idx = self._associate(trackable)

        for track_idx, det_idx in matches:
            track = self._tracks[track_idx]
            detection = trackable[det_idx]
            track.kalman.update(detection.bbox)
            track.time_since_update = 0
            track.hits += 1
            detection.track_id = track.track_id if track.hits >= self._min_hits else None
            self._maybe_update_embedding(track, detection, frame_bgr)

        for track_idx in unmatched_track_idx:
            self._tracks[track_idx].time_since_update += 1

        for det_idx in unmatched_det_idx:
            detection = trackable[det_idx]
            revived_id, embedding = self._try_revive(detection, frame_bgr)

            track_id = revived_id if revived_id is not None else self._next_id
            new_track = _Track(track_id=track_id, kalman=BoxKalmanFilter(detection.bbox))
            new_track.last_embedding = embedding
            detection.track_id = track_id if (revived_id is not None or self._min_hits <= 1) else None
            self._tracks.append(new_track)
            if revived_id is None:
                self._next_id += 1

        self._retire_stale_tracks()

        return trackable

    def _maybe_update_embedding(
        self, track: _Track, detection: PoseResult, frame_bgr: np.ndarray | None
    ) -> None:
        if self._embedder is None or frame_bgr is None:
            return
        embedding = self._embedder.embed(frame_bgr, detection.bbox)
        if embedding is not None:
            track.last_embedding = embedding

    def _try_revive(
        self, detection: PoseResult, frame_bgr: np.ndarray | None
    ) -> tuple[int | None, np.ndarray | None]:
        """For a detection that didn't IoU-match any active track, check it
        against the lost-track gallery by appearance. Returns
        (revived_track_id_or_None, computed_embedding_or_None) -- the
        embedding is computed (and returned) even when there's no gallery
        to check yet, so a brand-new track still gets a `last_embedding`
        to fall back on if it later ages out itself."""
        if self._embedder is None or frame_bgr is None:
            return None, None

        embedding = self._embedder.embed(frame_bgr, detection.bbox)
        if embedding is None or not self._lost_gallery:
            return None, embedding

        best_match: _LostTrack | None = None
        best_similarity = self._reid_similarity_threshold
        for lost in self._lost_gallery:
            similarity = cosine_similarity(embedding, lost.embedding)
            if similarity >= best_similarity:
                best_similarity = similarity
                best_match = lost

        if best_match is None:
            return None, embedding

        self._lost_gallery = [g for g in self._lost_gallery if g.track_id != best_match.track_id]
        return best_match.track_id, embedding

    def _retire_stale_tracks(self) -> None:
        still_active = []
        for track in self._tracks:
            if track.time_since_update <= self._max_age:
                still_active.append(track)
            elif self._embedder is not None and track.last_embedding is not None:
                self._lost_gallery.append(_LostTrack(track.track_id, track.last_embedding))
        self._tracks = still_active

        for lost in self._lost_gallery:
            lost.frames_lost += 1
        self._lost_gallery = [g for g in self._lost_gallery if g.frames_lost <= self._reid_gallery_ttl]

    def _associate(
        self, detections: list[PoseResult]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self._tracks or not detections:
            return [], list(range(len(self._tracks))), list(range(len(detections)))

        iou_matrix = np.zeros((len(self._tracks), len(detections)))
        for i, track in enumerate(self._tracks):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = _iou(track.predicted_box, detection.bbox)

        row_idx, col_idx = linear_sum_assignment(-iou_matrix)

        matches: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for r, c in zip(row_idx, col_idx):
            if iou_matrix[r, c] >= self._iou_threshold:
                matches.append((r, c))
                matched_tracks.add(r)
                matched_dets.add(c)

        unmatched_tracks = [i for i in range(len(self._tracks)) if i not in matched_tracks]
        unmatched_dets = [j for j in range(len(detections)) if j not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets
