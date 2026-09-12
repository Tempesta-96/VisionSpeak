"""Create and populate the isolated VisionSpeak worker environments.

The notebook kernel is expected to be the lip environment. This module can
pip-install the lip requirements into that interpreter when its imports are
missing, and create or populate the separate ``.venv-yolo`` worker environment
that ``inference.launcher`` discovers automatically.

All functions are standard-library only and never import OpenCV or Torch into
the caller's process; dependency probing happens in subprocesses.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

LIP_IMPORTS = ("cv2", "numpy", "torch", "mediapipe")
YOLO_IMPORTS = ("cv2", "numpy", "torch", "ultralytics")

IMPORT_PROBE_TIMEOUT_SECONDS = 120
INSTALL_TIMEOUT_SECONDS = 3600


def venv_python(environment_dir: Path) -> Path:
    """Return the interpreter path inside a virtual environment."""
    environment_dir = Path(environment_dir)
    if sys.platform == "win32":
        return environment_dir / "Scripts" / "python.exe"
    return environment_dir / "bin" / "python"


def environment_has_imports(python: Path, imports: tuple[str, ...]) -> bool:
    """Return True when ``python`` can import every requested module."""
    statement = "import " + ", ".join(imports)
    try:
        result = subprocess.run(
            [str(python), "-c", statement],
            capture_output=True,
            text=True,
            check=False,
            timeout=IMPORT_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def ensure_virtualenv(environment_dir: Path, *, force: bool = False) -> Path:
    """Return the environment interpreter, creating the venv when needed."""
    python = venv_python(environment_dir)
    if python.is_file() and not force:
        return python
    subprocess.run(
        [sys.executable, "-m", "venv", str(environment_dir)],
        check=True,
        timeout=INSTALL_TIMEOUT_SECONDS,
    )
    return python


def install_requirements(python: Path, requirements_path: Path) -> None:
    """Install a requirements file into ``python``, streaming output."""
    subprocess.run(
        [str(python), "-m", "pip", "install", "-r", str(requirements_path)],
        check=True,
        timeout=INSTALL_TIMEOUT_SECONDS,
    )


def setup_worker_environment(
    environment_dir: Path,
    requirements_path: Path,
    imports: tuple[str, ...] = YOLO_IMPORTS,
    *,
    force: bool = False,
) -> tuple[Path, bool]:
    """Create and populate an isolated worker environment.

    Returns ``(interpreter, installed)`` where ``installed`` reports whether
    pip ran during this call.
    """
    python = ensure_virtualenv(environment_dir, force=force)
    if not force and environment_has_imports(python, imports):
        return python, False
    install_requirements(python, requirements_path)
    return python, True


def setup_lip_environment(
    requirements_path: Path,
    imports: tuple[str, ...] = LIP_IMPORTS,
    *,
    force: bool = False,
) -> bool:
    """Install lip requirements into the current interpreter when needed.

    Returns True when pip ran, which means the notebook kernel must be
    restarted before the new packages can be imported.
    """
    if not force and environment_has_imports(Path(sys.executable), imports):
        return False
    install_requirements(Path(sys.executable), requirements_path)
    return True
