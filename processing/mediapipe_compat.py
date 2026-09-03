"""Import MediaPipe safely for vision-only environments.

Some MediaPipe builds eagerly import their optional audio recorder. On systems
without an active PortAudio server, importing ``sounddevice`` can raise before
the vision APIs become available. VisionSpeak does not use MediaPipe audio, so
we expose the dependency as unavailable when that optional initialization
fails, allowing MediaPipe's own ImportError fallback to handle it.
"""

from __future__ import annotations

import importlib
import sys


try:
    importlib.import_module("sounddevice")
except Exception:
    sys.modules.pop("sounddevice", None)
    sys.modules["sounddevice"] = None

mp = importlib.import_module("mediapipe")
python = importlib.import_module("mediapipe.tasks.python")
vision = importlib.import_module("mediapipe.tasks.python.vision")

__all__ = ["mp", "python", "vision"]
