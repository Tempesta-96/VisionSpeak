"""Subprocess launcher used by ``combinedVisionSpeak.ipynb``.

Lip reading and YOLO gesture detection intentionally run in different Python
environments because their third-party packages require different OpenCV wheel
distributions. Only one worker owns the webcam at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import pickletools
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from typing import Callable

from inference.protocol import EXIT_QUIT, EXIT_SWITCH, action_from_return_code


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = ("lip", "gesture")


class ConfigurationError(RuntimeError):
    """Raised when a combined-session dependency or path is unavailable."""


@dataclass(frozen=True)
class CombinedConfig:
    """User-configurable settings for a combined VisionSpeak session."""

    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    initial_mode: str = "lip"
    camera_index: int = 0
    lip_python: str | Path | None = None
    gesture_python: str | Path | None = None
    lip_checkpoint: str | Path = Path(
        "checkpoints/valente_custom_words_lipnet_ctc_0.852_acc.pt"
    )
    gesture_checkpoint: str | Path = Path("checkpoints/yolov8n_custom.pt")
    countdown_seconds: float = 2.0
    capture_seconds: float = 3.0
    min_frames: int = 30
    min_detection_rate: float = 0.75
    max_frames: int = 90
    gesture_confidence: float = 0.5
    lip_device: str = "auto"
    gesture_device: str | None = None


@dataclass(frozen=True)
class ResolvedConfig:
    """Validated configuration with absolute filesystem paths."""

    project_root: Path
    initial_mode: str
    camera_index: int
    lip_python: Path
    gesture_python: Path
    lip_checkpoint: Path
    gesture_checkpoint: Path
    countdown_seconds: float
    capture_seconds: float
    min_frames: int
    min_detection_rate: float
    max_frames: int
    gesture_confidence: float
    lip_device: str
    gesture_device: str | None
    gesture_backend: str = "yolo"


@dataclass(frozen=True)
class SessionEvent:
    """One completed worker invocation."""

    mode: str
    action: str
    return_code: int
    elapsed_seconds: float


@dataclass(frozen=True)
class SessionSummary:
    """Result returned to the notebook after the webcam session ends."""

    initial_mode: str
    events: tuple[SessionEvent, ...]
    error: str | None = None

    @property
    def completed(self) -> bool:
        return self.error is None

    @property
    def switch_count(self) -> int:
        return sum(event.action == "switch" for event in self.events)


def _absolute_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _absolute_interpreter_path(value: str | Path, project_root: Path) -> Path:
    """Make an interpreter path absolute without resolving its venv symlink."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return Path(os.path.abspath(path))


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        absolute = Path(os.path.abspath(path.expanduser()))
        if absolute not in seen:
            seen.add(absolute)
            unique.append(absolute)
    return unique


def _interpreter_candidates(
    mode: str,
    configured: str | Path | None,
    project_root: Path,
) -> list[Path]:
    if configured is not None:
        return [_absolute_interpreter_path(configured, project_root)]

    environment_name = f"VISIONSPEAK_{mode.upper()}_PYTHON"
    environment_value = os.environ.get(environment_name)
    candidates: list[Path] = []
    if environment_value:
        candidates.append(_absolute_interpreter_path(environment_value, project_root))

    environment_directory = ".venv" if mode == "lip" else ".venv-yolo"
    candidates.extend(
        [
            project_root / environment_directory / "bin" / "python",
            project_root / environment_directory / "Scripts" / "python.exe",
        ]
    )
    if mode == "lip":
        candidates.append(Path(sys.executable))
    return _unique_paths(candidates)


def resolve_interpreter(
    mode: str,
    configured: str | Path | None,
    project_root: Path,
) -> Path:
    """Resolve the Python executable for one worker environment."""
    candidates = _interpreter_candidates(mode, configured, project_root)
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n  - ".join(str(path) for path in candidates)
    requirements_name = "requirements.txt" if mode == "lip" else "requirements-yolo.txt"
    raise ConfigurationError(
        f"Could not find the {mode} Python environment. Searched:\n  - {searched}\n"
        f"Create it from {requirements_name}, or set the {mode}_python notebook option."
    )


def _check_environment_imports(mode: str, interpreter: Path, project_root: Path) -> None:
    if mode == "lip":
        imports = (
            "import importlib.metadata as metadata; "
            "names={d.metadata['Name'].lower() for d in metadata.distributions() if d.metadata['Name']}; "
            "assert 'opencv-contrib-python' in names and 'opencv-python' not in names, "
            "'lip environment must contain only opencv-contrib-python'; "
            "import cv2, numpy, torch; from processing.mediapipe_compat import mp"
        )
    else:
        imports = (
            "import importlib.metadata as metadata; "
            "names={d.metadata['Name'].lower() for d in metadata.distributions() if d.metadata['Name']}; "
            "assert 'opencv-python' in names and 'opencv-contrib-python' not in names, "
            "'YOLO environment must contain only opencv-python'; "
            "import cv2, numpy, torch, ultralytics"
        )
    try:
        result = subprocess.run(
            [str(interpreter), "-c", imports],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ConfigurationError(
            f"Could not validate the {mode} environment at {interpreter}: {error}"
        ) from error
    if result.returncode:
        details = (result.stderr or result.stdout).strip()
        requirements_name = "requirements.txt" if mode == "lip" else "requirements-yolo.txt"
        raise ConfigurationError(
            f"The {mode} environment failed its import or OpenCV-isolation check. "
            f"Recreate it and install {requirements_name} with {interpreter}.\n{details}"
        )


def detect_gesture_backend(checkpoint: Path) -> str:
    """Inspect Torch serialization metadata without executing pickle globals.

    YOLO stores an Ultralytics model object; SX stores a state dictionary and
    class mapping. Tensor data is not read and filenames do not select models.
    """
    try:
        with zipfile.ZipFile(checkpoint) as archive:
            entries = [name for name in archive.namelist() if name.endswith("/data.pkl") or name == "data.pkl"]
            if len(entries) != 1:
                raise ValueError("Expected one Torch data.pkl metadata entry.")
            strings = set()
            for opcode, argument, _ in pickletools.genops(archive.read(entries[0])):
                if isinstance(argument, str) and opcode.name in (
                    "UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8", "GLOBAL",
                ):
                    strings.add(argument)
        sx = {"class_to_idx", "model_state_dict"}.issubset(strings)
        yolo = (bool({"model", "ema"}.intersection(strings))
                and any(value.startswith("ultralytics.") for value in strings))
        if sx != yolo:
            return "sx" if sx else "yolo"
        raise ValueError("Expected an SX class_to_idx/state-dict checkpoint or an Ultralytics model checkpoint.")
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ConfigurationError(f"Cannot identify gesture checkpoint {checkpoint.name}: {error}") from error


def validate_combined_config(
    config: CombinedConfig,
    *,
    check_imports: bool = True,
) -> ResolvedConfig:
    """Validate paths, numeric settings, and optionally third-party imports."""
    project_root = Path(config.project_root).expanduser().resolve()
    if config.initial_mode not in MODES:
        raise ConfigurationError(f"initial_mode must be one of {MODES}.")
    if config.camera_index < 0:
        raise ConfigurationError("camera_index must be zero or greater.")
    if config.countdown_seconds < 0 or config.capture_seconds <= 0:
        raise ConfigurationError("Countdown must be non-negative and capture time positive.")
    if config.min_frames <= 0 or config.max_frames < config.min_frames:
        raise ConfigurationError("Frame limits must be positive and max_frames >= min_frames.")
    if not 0 <= config.min_detection_rate <= 1:
        raise ConfigurationError("min_detection_rate must be between zero and one.")
    if not 0 <= config.gesture_confidence <= 1:
        raise ConfigurationError("gesture_confidence must be between zero and one.")

    lip_python = resolve_interpreter("lip", config.lip_python, project_root)
    lip_checkpoint = _absolute_path(config.lip_checkpoint, project_root)
    gesture_checkpoint = _absolute_path(config.gesture_checkpoint, project_root)

    missing = [path for path in (lip_checkpoint, gesture_checkpoint) if not path.is_file()]
    if missing:
        raise ConfigurationError(
            "Missing model checkpoint(s):\n  - " + "\n  - ".join(str(path) for path in missing)
        )

    gesture_backend = detect_gesture_backend(gesture_checkpoint)
    gesture_python = (lip_python if gesture_backend == "sx" else
                      resolve_interpreter("gesture", config.gesture_python, project_root))

    if check_imports:
        _check_environment_imports("lip", lip_python, project_root)
        if gesture_backend == "yolo":
            _check_environment_imports("gesture", gesture_python, project_root)
        else:
            try:
                result = subprocess.run(
                    [str(lip_python), "-m", "inference.sx_worker", "--checkpoint",
                     str(gesture_checkpoint), "--device", "cpu", "--check-only"],
                    cwd=project_root, capture_output=True, text=True, timeout=60,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise ConfigurationError(f"Could not validate the SX checkpoint: {error}") from error
            if result.returncode:
                raise ConfigurationError("SX checkpoint or Tasks detector validation failed:\n" +
                                         (result.stderr or result.stdout).strip())

    return ResolvedConfig(
        project_root=project_root,
        initial_mode=config.initial_mode,
        camera_index=config.camera_index,
        lip_python=lip_python,
        gesture_python=gesture_python,
        lip_checkpoint=lip_checkpoint,
        gesture_checkpoint=gesture_checkpoint,
        countdown_seconds=config.countdown_seconds,
        capture_seconds=config.capture_seconds,
        min_frames=config.min_frames,
        min_detection_rate=config.min_detection_rate,
        max_frames=config.max_frames,
        gesture_confidence=config.gesture_confidence,
        lip_device=config.lip_device,
        gesture_device=config.gesture_device,
        gesture_backend=gesture_backend,
    )


def build_worker_command(mode: str, config: ResolvedConfig) -> list[str]:
    """Build the command for one isolated webcam worker."""
    if mode == "lip":
        return [
            str(config.lip_python),
            "-m",
            "inference.lip_worker",
            "--camera-index",
            str(config.camera_index),
            "--checkpoint",
            str(config.lip_checkpoint),
            "--countdown-seconds",
            str(config.countdown_seconds),
            "--capture-seconds",
            str(config.capture_seconds),
            "--min-frames",
            str(config.min_frames),
            "--min-detection-rate",
            str(config.min_detection_rate),
            "--max-frames",
            str(config.max_frames),
            "--device",
            config.lip_device,
        ]

    command = [
        str(config.gesture_python),
        "-m",
        "inference.sx_worker" if config.gesture_backend == "sx" else "inference.gesture_worker",
        "--camera-index",
        str(config.camera_index),
        "--checkpoint",
        str(config.gesture_checkpoint),
        "--confidence",
        str(config.gesture_confidence),
    ]
    if config.gesture_device:
        command.extend(["--device", config.gesture_device])
    return command


def _run_worker(mode: str, config: ResolvedConfig) -> int:
    process = subprocess.Popen(build_worker_command(mode, config), cwd=config.project_root)
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return EXIT_QUIT


WorkerRunner = Callable[[str, ResolvedConfig], int]


def run_combined_session(
    config: CombinedConfig,
    *,
    worker_runner: WorkerRunner | None = None,
) -> SessionSummary:
    """Run workers until the user quits or one worker fails."""
    resolved = validate_combined_config(config, check_imports=False)
    runner = worker_runner or _run_worker
    current_mode = resolved.initial_mode
    events: list[SessionEvent] = []

    while True:
        print(f"Starting {current_mode} mode. Press M to switch or Q to quit.")
        started = time.monotonic()
        try:
            return_code = runner(current_mode, resolved)
        except Exception as error:
            message = f"Unable to run the {current_mode} worker: {error}"
            events.append(
                SessionEvent(
                    mode=current_mode,
                    action="error",
                    return_code=EXIT_QUIT + 1,
                    elapsed_seconds=time.monotonic() - started,
                )
            )
            return SessionSummary(resolved.initial_mode, tuple(events), message)

        action = action_from_return_code(return_code)
        events.append(
            SessionEvent(
                mode=current_mode,
                action=action.value,
                return_code=return_code,
                elapsed_seconds=time.monotonic() - started,
            )
        )
        if return_code == EXIT_SWITCH:
            current_mode = "gesture" if current_mode == "lip" else "lip"
            continue
        if return_code == EXIT_QUIT:
            return SessionSummary(resolved.initial_mode, tuple(events))
        return SessionSummary(
            resolved.initial_mode,
            tuple(events),
            f"The {current_mode} worker stopped unexpectedly with exit code {return_code}.",
        )
