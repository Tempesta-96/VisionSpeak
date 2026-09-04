"""Shared process-exit protocol for the combined VisionSpeak session."""

from __future__ import annotations

from enum import Enum


EXIT_QUIT = 0
EXIT_ERROR = 1
EXIT_SWITCH = 10


class WorkerAction(str, Enum):
    """Actions a webcam worker can return to the notebook launcher."""

    QUIT = "quit"
    SWITCH = "switch"
    ERROR = "error"


def action_from_return_code(return_code: int) -> WorkerAction:
    """Translate a worker return code into a launcher action."""
    if return_code == EXIT_QUIT:
        return WorkerAction.QUIT
    if return_code == EXIT_SWITCH:
        return WorkerAction.SWITCH
    return WorkerAction.ERROR
