"""Combined VisionSpeak webcam CLI.

Terminal equivalent of ``combinedVisionSpeak.ipynb``: one webcam session that
switches between the custom eight-word CTC lip reader and the YOLO gesture
detector, each running in its own isolated Python environment.

Run this file with the lip environment because that interpreter is the
launcher's fallback for the lip worker::

    conda run -n visionspeak python main.py
    .venv/bin/python main.py

Use ``--check-only`` to validate both environments and checkpoints without
opening the webcam. If the lip environment is not the current interpreter,
pass its path explicitly with ``--lip-python``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from inference.launcher import (
    CombinedConfig,
    ConfigurationError,
    ResolvedConfig,
    SessionSummary,
    run_combined_session,
    validate_combined_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LIP_CHECKPOINT = Path(
    "checkpoints/valente_custom_words_lipnet_ctc_0.852_acc.pt"
)
DEFAULT_GESTURE_CHECKPOINT = Path("checkpoints/yolov8n_custom.pt")

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser mirroring the ``CombinedConfig`` fields."""
    parser = argparse.ArgumentParser(
        description=(
            "Run one combined VisionSpeak webcam session with the CTC lip "
            "reader and the YOLO gesture detector."
        ),
        epilog=(
            "OpenCV window controls: SPACE records one lip word, M switches "
            "mode, Q quits."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("lip", "gesture"),
        default="lip",
        help="Initial session mode (default: lip).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV webcam index (default: 0).",
    )
    parser.add_argument(
        "--lip-python",
        type=Path,
        default=None,
        help=(
            "Interpreter for the lip environment. Defaults to auto-discovery "
            "(.venv or the current interpreter)."
        ),
    )
    parser.add_argument(
        "--gesture-python",
        type=Path,
        default=None,
        help=(
            "Interpreter for the YOLO environment. Defaults to auto-discovery "
            "(.venv-yolo)."
        ),
    )
    parser.add_argument(
        "--lip-checkpoint",
        type=Path,
        default=DEFAULT_LIP_CHECKPOINT,
        help=f"Lip checkpoint (default: {DEFAULT_LIP_CHECKPOINT}).",
    )
    parser.add_argument(
        "--gesture-checkpoint",
        type=Path,
        default=DEFAULT_GESTURE_CHECKPOINT,
        help=f"YOLO checkpoint (default: {DEFAULT_GESTURE_CHECKPOINT}).",
    )
    parser.add_argument(
        "--countdown-seconds",
        type=float,
        default=2.0,
        help="Countdown before lip recording (default: 2.0).",
    )
    parser.add_argument(
        "--capture-seconds",
        type=float,
        default=3.0,
        help="Lip recording duration (default: 3.0).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=30,
        help="Minimum captured mouth frames (default: 30).",
    )
    parser.add_argument(
        "--min-detection-rate",
        type=float,
        default=0.75,
        help="Minimum fresh mouth-detection rate, 0-1 (default: 0.75).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=90,
        help="Maximum frames passed to the CTC model (default: 90).",
    )
    parser.add_argument(
        "--gesture-confidence",
        type=float,
        default=0.5,
        help="YOLO detection confidence, 0-1 (default: 0.5).",
    )
    parser.add_argument(
        "--lip-device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device for the lip model (default: auto).",
    )
    parser.add_argument(
        "--gesture-device",
        default=None,
        help="Device for Ultralytics; the default lets it choose.",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip the subprocess import checks for both environments.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Validate environments and checkpoints, then exit without opening "
            "the webcam."
        ),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> CombinedConfig:
    """Translate parsed CLI arguments into a ``CombinedConfig``."""
    return CombinedConfig(
        project_root=PROJECT_ROOT,
        initial_mode=args.mode,
        camera_index=args.camera_index,
        lip_python=args.lip_python,
        gesture_python=args.gesture_python,
        lip_checkpoint=args.lip_checkpoint,
        gesture_checkpoint=args.gesture_checkpoint,
        countdown_seconds=args.countdown_seconds,
        capture_seconds=args.capture_seconds,
        min_frames=args.min_frames,
        min_detection_rate=args.min_detection_rate,
        max_frames=args.max_frames,
        gesture_confidence=args.gesture_confidence,
        lip_device=args.lip_device,
        gesture_device=args.gesture_device,
    )


def print_resolved(resolved: ResolvedConfig) -> None:
    """Print the validated environments and checkpoints (notebook cell 3)."""
    print(f"Project root:    {resolved.project_root}")
    print(f"Lip Python:      {resolved.lip_python}")
    print(f"Gesture Python:  {resolved.gesture_python}")
    print(f"Lip checkpoint:  {resolved.lip_checkpoint.name}")
    print(f"YOLO checkpoint: {resolved.gesture_checkpoint.name}")


def print_summary(summary: SessionSummary) -> None:
    """Print the session outcome (notebook cell 5)."""
    status = "Completed" if summary.completed else "Stopped with an error"
    print(status)
    print(f"Initial mode: {summary.initial_mode}")
    print(f"Mode switches: {summary.switch_count}")
    for number, event in enumerate(summary.events, start=1):
        print(
            f"{number}. {event.mode}: {event.action} "
            f"(exit {event.return_code}, {event.elapsed_seconds:.1f}s)"
        )
    if summary.error:
        print(f"Error: {summary.error}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Run the combined CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

    try:
        resolved = validate_combined_config(
            config,
            check_imports=not args.skip_checks,
        )
    except ConfigurationError as error:
        print(f"Configuration error:\n{error}", file=sys.stderr)
        return EXIT_ERROR

    print_resolved(resolved)
    if args.check_only:
        return EXIT_SUCCESS

    try:
        summary = run_combined_session(config)
    except KeyboardInterrupt:
        print("Session interrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED

    print_summary(summary)
    return EXIT_SUCCESS if summary.completed else EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
