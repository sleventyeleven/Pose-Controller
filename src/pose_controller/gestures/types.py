from __future__ import annotations

from enum import Enum, auto


class ArmPose(Enum):
    DOWN = auto()  # neutral -- arm at rest
    RAISED = auto()  # wrist well above shoulder (overhead)
    OUT_TO_SIDE = auto()  # wrist extended away from torso at roughly shoulder height


class ControlAction(Enum):
    """Abstract media control intents emitted by gesture recognition.

    Deliberately not tied to Spotify or any specific player -- that's
    `control/`'s job (milestone 6, not yet built). Gesture logic only
    decides *that* an action was triggered, not how it gets carried out.
    """

    NEXT = auto()
    PREVIOUS = auto()
    SKIP = auto()
    PLAY_PAUSE = auto()
