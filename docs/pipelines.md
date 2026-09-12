# Pipeline reference: design and performance, all three backends

This is the live current-state reference for `inference.backend`'s three
options -- what each pipeline actually does, stage by stage, and every
real performance number measured for it so far, on every board tested.
`docs/backlog.md` has the investigative narrative behind each of these
(what was tried, what failed, why a given design decision was made);
this document is just "what's true right now," kept in one place instead
of scattered across several backlog entries that would otherwise drift
out of sync with each other. `docs/journey.md` has the story-length
version for context on how the project got here at all.

## `qnn` -- YOLOv8n-det + BlazePose-landmark (current default)

```mermaid
flowchart TB
    A[Capture: BGR frame] --> B["YOLOv8n-det<br/>640x640, w8a8<br/>(models/yolov8n_det_qcs6490)"]
    B -->|"person boxes<br/>(NMS + person-class filter)"| C["Per-box crop/ROI<br/>_roi_corners_from_box<br/>(BOX_ROI_SCALE=1.75)"]
    C --> D["BlazePose landmark<br/>256x256, w8a8<br/>(models/mediapipe_pose_qcs6490)"]
    D -->|"25 landmarks (x,y,z,vis)<br/>+ one pose-level confidence"| E{"score >=<br/>MIN_LANDMARK_SCORE?"}
    E -->|no| F[pose dropped entirely]
    E -->|yes| G["Map to Landmark enum<br/>-> PoseResult"]
    G --> H["Tracker<br/>(Kalman+IoU, OSNet re-ID)"]
    H --> I[GestureStateMachine]
    I --> J["overlay/ + dashboard +<br/>MediaController.dispatch"]
```

Two BlazePose-specific weak points motivated evaluating the alternatives
below: the crop/ROI step is a whole extra layer that's already needed
real tuning (`BOX_ROI_SCALE`, frame-edge clamping -- see
`scripts/qnn_spike.md`), and the single pose-level `MIN_LANDMARK_SCORE`
gate at **E** rejects the *entire* pose if it dips, even when most
individual keypoints are perfectly visible.

## `hrnet` -- YOLOv8n-det + HRNetPose-landmark (alongside evaluation)

```mermaid
flowchart TB
    A[Capture: BGR frame] --> B["YOLOv8n-det<br/>640x640, w8a8<br/>(models/yolov8n_det_qcs6490)"]
    B -->|"person boxes<br/>(NMS + person-class filter)"| C["Per-box crop/ROI<br/>_roi_corners_from_box<br/>(BOX_ROI_SCALE=1.75, shared with qnn)"]
    C --> D["HRNetPose landmark<br/>256x192, w8a8<br/>(models/hrnet_pose_qcs6490)"]
    D -->|"64x48x17 heatmap<br/>(one confidence map per keypoint)"| E["decode_heatmaps:<br/>per-channel argmax + x4 stride"]
    E -->|"17 keypoints, each with its<br/>own independent visibility"| G["Map to Landmark enum<br/>-> PoseResult"]
    G --> H["Tracker<br/>(Kalman+IoU, OSNet re-ID)"]
    H --> I[GestureStateMachine]
    I --> J["overlay/ + dashboard +<br/>MediaController.dispatch"]
```

Same first stage and crop geometry as `qnn` -- only the landmark model
changes. No single whole-pose confidence gate: each keypoint's own
heatmap peak is its confidence, so a low-confidence wrist doesn't take
down an otherwise-good shoulder. Uses the same proven `w8a8` quantization
path as every other model here, unlike `yolo26` below.

## `yolo26` -- YOLO26-Pose, single end-to-end model (alongside evaluation)

```mermaid
flowchart TB
    A[Capture: BGR frame] --> B["YOLO26-Pose<br/>640x640, w8a16<br/>(models/yolo26n_pose_qcs6490)"]
    B -->|"[56, 8400] raw per-anchor output:<br/>box(cxcywh) + person_conf + 17 keypoints"| C["decode_raw_output:<br/>cxcywh -> xyxy, channel split"]
    C --> D{"person_conf >=<br/>NMS_SCORE_THRESHOLD?"}
    D -->|no| F[anchor dropped]
    D -->|yes| E["Greedy NMS across<br/>surviving anchors"]
    E -->|"box + 17 keypoints,<br/>each with its own visibility"| G["Map COCO keypoints to<br/>Landmark enum -> PoseResult"]
    G --> H["Tracker<br/>(Kalman+IoU, OSNet re-ID)"]
    H --> I[GestureStateMachine]
    I --> J["overlay/ + dashboard +<br/>MediaController.dispatch"]
```

No crop step at all -- detection and keypoints come out of one forward
pass over the whole frame, and (like `hrnet`) no single whole-pose
confidence gate. The tradeoff for removing the crop step: a full
640x640 forward pass every frame instead of a small per-person crop, which
shows up directly in the performance numbers below. Model compiled via
Ultralytics' own local QNN export, not Qualcomm AI Hub (whose compiler
fails on this model) -- see `docs/backlog.md` for that whole story,
including why the export must target QCS6490's real `soc_model` (`93`)
rather than a generic Hexagon-architecture-version target.

## Performance: every stage, every board tested

All NPU numbers below are median warm inference time over 20 calls, via
the exact `onnxruntime-qnn` runtime pattern used in production
(`inference/backends/_qnn_runtime.py`'s `create_qnn_session`), measured
directly over SSH on each physical board -- not simulator/estimate
numbers, except where noted.

| Stage | Model | Q6A (QCS6490) | Q8B (SC8280XP) |
|---|---|---|---|
| Person detector (`qnn` + `hrnet`'s shared 1st stage) | `yolov8n_det_qcs6490` | ~2.2ms | ~5.9ms |
| Landmark (`qnn` 2nd stage) | `mediapipe_pose_qcs6490` (landmark) | ~1.1ms | ~1.0ms |
| Landmark (`hrnet` 2nd stage) | `hrnet_pose_qcs6490` | ~5.7ms | ~6.5ms |
| End-to-end detect+pose (`yolo26`) | `yolo26n_pose_qcs6490` | ~14.9ms | ~9.9ms |
| General 80-class detect (dashboard viz only, not the pose pipeline) | `yolo26_det_qcs6490` | ~14.7ms | ~9.6ms |

Note the Q6A/Q8B relationship flips between rows: the Q8B is slower for
the two-stage pipeline's models but faster for the YOLO26 models,
despite sharing the same Hexagon V68 architecture. Not root-caused --
plausible clock/binning/driver-stack differences between the two
physical boards -- but consistent enough across repeated runs that it's
a real effect, not measurement noise.

## Full-pipeline real-footage results

Measured running the *entire* pipeline (capture -> pose -> track ->
gesture, identical `Tracker`/`ReidEmbedder` config) against the same
recorded footage on the Q6A -- see `docs/backlog.md` for the full
writeup and caveats (single person in frame, small YOLO26 calibration
set).

| Pipeline | Frames with a pose detected | Track IDs for the one real person | Gestures triggered | Speed |
|---|---|---|---|---|
| `qnn` (default) | 451/923 (49%) | 2 (fragmented) | 7 | ~22.1 fps |
| `yolo26` | 890/923 (96%) | 1 (continuous) | 14 | ~12.6 fps |
| `hrnet` | not yet tested | not yet tested | not yet tested | not yet tested |

The `yolo26` fps gap versus `qnn` isn't yet root-caused -- see
`docs/backlog.md`'s Performance section for the current hypotheses
(compute cost of a full 640x640 forward pass vs. a small crop, `w8a16`
vs. `w8a8` overhead, Python-side decode cost) and what would need
measuring to tell them apart.
