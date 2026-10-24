"""YOLO webcam worker for the combined VisionSpeak notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

from inference.protocol import EXIT_ERROR, EXIT_QUIT, EXIT_SWITCH


def run_live(detector) -> str:
    """Add switch-aware live controls around the unchanged GestureDetector."""
    if not detector.open_camera():
        detector.close_camera()
        return "error"

    print("Gesture mode active. Press M for lip mode or Q to quit.")
    try:
        while True:
            success, frame = detector.camera.read()
            if not success:
                print("Error: Could not read frame from camera")
                return "error"

            frame = cv2.flip(frame, 1)
            gesture_name, confidence, result_frame = detector.predict_frame(frame)
            cv2.rectangle(result_frame, (10, 8), (410, 48), (24, 24, 24), -1)
            cv2.putText(
                result_frame,
                "GESTURE MODE | M lip | Q quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(
                result_frame,
                (10, result_frame.shape[0] - 48),
                (350, result_frame.shape[0] - 8),
                (24, 24, 24),
                -1,
            )
            cv2.putText(
                result_frame,
                f"{gesture_name} ({confidence:.1%})",
                (20, result_frame.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(detector.window_name, result_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("m"), ord("M")):
                return "switch"
            if key in (ord("q"), ord("Q")):
                return "quit"
    finally:
        detector.close_camera()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VisionSpeak YOLO gesture worker.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from models.Lawrence.gesture_detector import GestureDetector

        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Gesture checkpoint not found: {args.checkpoint}")
        detector = GestureDetector(
            str(args.checkpoint),
            camera_index=args.camera_index,
            confidence=args.confidence,
        )
        if args.device is not None:
            detector.model.to(args.device)
        action = run_live(detector)
        if action == "switch":
            return EXIT_SWITCH
        if action == "quit":
            return EXIT_QUIT
        return EXIT_ERROR
    except Exception as error:
        print(f"Gesture worker error: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
