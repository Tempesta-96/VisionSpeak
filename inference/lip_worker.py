"""MediaPipe + CTC webcam worker for the combined VisionSpeak notebook."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from inference.protocol import EXIT_ERROR, EXIT_QUIT, EXIT_SWITCH
from models.valente.model import LipNet
from processing.mouth_cropper import MouthCropper


WORDS = ("hello", "hi", "yes", "no", "good", "bad", "thanks", "bye")
FRAME_HEIGHT = 50
FRAME_WIDTH = 100
WINDOW_NAME = "VisionSpeak - Lip Reading"


@dataclass(frozen=True)
class LipPrediction:
    label: str
    confidence: float
    scores: dict[str, float]


@dataclass(frozen=True)
class CaptureResult:
    action: str
    frames: np.ndarray | None = None
    detection_rate: float = 0.0
    message: str = ""


class TimestampClock:
    """Generate strictly increasing MediaPipe video timestamps."""

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.last_timestamp_ms = -1

    def next(self) -> int:
        timestamp = int((time.monotonic() - self.started) * 1000)
        timestamp = max(timestamp, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp
        return timestamp


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_clip(
    frames: np.ndarray,
    *,
    max_frames: int = 90,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare uint8 mouth frames exactly as the custom CTC training notebook."""
    source = np.asarray(frames)
    if source.dtype != np.uint8 or source.ndim != 3:
        raise ValueError("frames must be uint8 with shape (time, 50, 100).")
    if len(source) == 0 or source.shape[1:] != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError("frames must be non-empty with shape (time, 50, 100).")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive.")

    if len(source) > max_frames:
        indices = np.linspace(0, len(source) - 1, max_frames).round().astype(int)
        source = source[indices]

    length = len(source)
    normalized = source.astype(np.float32) / 255.0
    normalized = (normalized - normalized.mean()) / (normalized.std() + 1e-6)
    padded = np.zeros((max_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)
    padded[:length] = normalized
    tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(2)
    lengths = torch.tensor([length], dtype=torch.long)
    return tensor, lengths


def score_vocabulary(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    words: tuple[str, ...],
    char_to_index: dict[str, int],
    *,
    blank_index: int = 0,
) -> dict[str, float]:
    """Return normalized CTC sequence-likelihood scores for allowed words."""
    if log_probs.ndim != 3 or log_probs.shape[1] != 1:
        raise ValueError("log_probs must have shape (time, 1, classes).")
    if not words:
        raise ValueError("At least one vocabulary word is required.")

    cpu_log_probs = log_probs.detach().to("cpu")
    cpu_input_lengths = input_lengths.detach().to(dtype=torch.long, device="cpu")
    sequence_scores: list[torch.Tensor] = []
    for word in words:
        missing = sorted(set(word).difference(char_to_index))
        if missing:
            raise ValueError(f"Checkpoint vocabulary cannot encode {word!r}: {missing}")
        target = torch.tensor([char_to_index[character] for character in word], dtype=torch.long)
        target_lengths = torch.tensor([len(target)], dtype=torch.long)
        loss = F.ctc_loss(
            cpu_log_probs,
            target,
            cpu_input_lengths,
            target_lengths,
            blank=blank_index,
            reduction="sum",
            zero_infinity=False,
        )
        sequence_scores.append(-loss)

    stacked_scores = torch.stack(sequence_scores)
    if not torch.isfinite(stacked_scores).any():
        raise ValueError("No vocabulary word has a finite CTC score.")
    probabilities = torch.softmax(stacked_scores, dim=0)
    return {word: float(probability) for word, probability in zip(words, probabilities)}


def clip_quality_message(
    frame_count: int,
    detection_rate: float,
    *,
    min_frames: int,
    min_detection_rate: float,
) -> str | None:
    """Return a retry message when a captured mouth clip is unreliable."""
    if frame_count < min_frames:
        return f"Retry: captured {frame_count} frames; at least {min_frames} are required."
    if detection_rate < min_detection_rate:
        return (
            f"Retry: mouth detected in {detection_rate:.0%} of frames; "
            f"at least {min_detection_rate:.0%} is required."
        )
    return None


class LipPredictor:
    """Load the custom CTC checkpoint and score the fixed word vocabulary."""

    def __init__(self, checkpoint_path: Path, device_name: str = "auto") -> None:
        self.device = choose_device(device_name)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.char_to_index = {str(key): int(value) for key, value in checkpoint["char_to_index"].items()}
        for word in WORDS:
            missing = set(word).difference(self.char_to_index)
            if missing:
                raise ValueError(f"Checkpoint cannot encode {word!r}: {sorted(missing)}")

        self.model = LipNet(
            num_classes=len(self.char_to_index) + 1,
            input_channels=1,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def predict(self, frames: np.ndarray, *, max_frames: int = 90) -> LipPrediction:
        inputs, lengths = prepare_clip(frames, max_frames=max_frames)
        log_probs, output_lengths = self.model.ctc_log_probs(
            inputs.to(self.device),
            lengths,
        )
        scores = score_vocabulary(log_probs, output_lengths, WORDS, self.char_to_index)
        label = max(scores, key=scores.get)
        return LipPrediction(label=label, confidence=scores[label], scores=scores)


def _key_action(key: int) -> str | None:
    key &= 0xFF
    if key in (ord("q"), ord("Q")):
        return "quit"
    if key in (ord("m"), ord("M")):
        return "switch"
    if key == ord(" "):
        return "capture"
    return None


def _annotate(
    frame: np.ndarray,
    box: tuple[int, int, int, int] | None,
    fresh_detection: bool,
    status: str,
) -> np.ndarray:
    preview = frame.copy()
    if box is None:
        cv2.putText(
            preview,
            "Mouth not detected",
            (20, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        colour = (0, 255, 0) if fresh_detection else (0, 165, 255)
        x1, y1, x2, y2 = box
        cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), colour, 2)

    cv2.putText(
        preview,
        "LIP MODE | SPACE record | M gesture | Q quit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        status[:90],
        (20, preview.shape[0] - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return preview


def _read_mouth_frame(
    camera,
    cropper: MouthCropper,
    clock: TimestampClock,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int] | None, bool]:
    success, frame = camera.read()
    if not success:
        raise RuntimeError("The webcam stopped returning frames.")
    crop, box, fresh_detection = cropper.crop_with_metadata(frame, clock.next())
    grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.uint8, copy=False)
    return frame, grayscale, box, fresh_detection


def capture_clip(
    camera,
    cropper: MouthCropper,
    clock: TimestampClock,
    *,
    countdown_seconds: float,
    capture_seconds: float,
    min_frames: int,
    min_detection_rate: float,
) -> CaptureResult:
    """Capture one fixed-duration clip while keeping quit/switch controls active."""
    cropper.reset_tracking()
    countdown_started = time.monotonic()
    while time.monotonic() - countdown_started < countdown_seconds:
        frame, _, box, fresh = _read_mouth_frame(camera, cropper, clock)
        remaining = max(0.0, countdown_seconds - (time.monotonic() - countdown_started))
        cv2.imshow(WINDOW_NAME, _annotate(frame, box, fresh, f"Get ready: {remaining:.1f}s"))
        action = _key_action(cv2.waitKey(1))
        if action in {"quit", "switch"}:
            return CaptureResult(action=action)

    frames: list[np.ndarray] = []
    fresh_detections = 0
    capture_started = time.monotonic()
    while time.monotonic() - capture_started < capture_seconds:
        frame, crop, box, fresh = _read_mouth_frame(camera, cropper, clock)
        frames.append(crop)
        fresh_detections += int(fresh)
        elapsed = time.monotonic() - capture_started
        progress = min(elapsed / capture_seconds, 1.0)
        cv2.imshow(
            WINDOW_NAME,
            _annotate(frame, box, fresh, f"Recording: {progress:.0%}"),
        )
        action = _key_action(cv2.waitKey(1))
        if action in {"quit", "switch"}:
            return CaptureResult(action=action)

    detection_rate = fresh_detections / max(len(frames), 1)
    retry_message = clip_quality_message(
        len(frames),
        detection_rate,
        min_frames=min_frames,
        min_detection_rate=min_detection_rate,
    )
    if retry_message:
        return CaptureResult(
            action="retry",
            detection_rate=detection_rate,
            message=retry_message,
        )
    return CaptureResult(
        action="predict",
        frames=np.stack(frames).astype(np.uint8, copy=False),
        detection_rate=detection_rate,
    )


def run_lip_webcam(
    predictor: LipPredictor,
    *,
    camera_index: int,
    countdown_seconds: float,
    capture_seconds: float,
    min_frames: int,
    min_detection_rate: float,
    max_frames: int,
) -> str:
    """Run the interactive isolated-word webcam loop."""
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Unable to open webcam index {camera_index}.")

    status = "Ready to record one word."
    clock = TimestampClock()
    try:
        with MouthCropper(target_size=(FRAME_WIDTH, FRAME_HEIGHT)) as cropper:
            while True:
                frame, _, box, fresh = _read_mouth_frame(camera, cropper, clock)
                cv2.imshow(WINDOW_NAME, _annotate(frame, box, fresh, status))
                action = _key_action(cv2.waitKey(1))
                if action in {"quit", "switch"}:
                    return action
                if action != "capture":
                    continue

                result = capture_clip(
                    camera,
                    cropper,
                    clock,
                    countdown_seconds=countdown_seconds,
                    capture_seconds=capture_seconds,
                    min_frames=min_frames,
                    min_detection_rate=min_detection_rate,
                )
                if result.action in {"quit", "switch"}:
                    return result.action
                if result.action == "retry":
                    status = result.message
                    print(status)
                    continue

                status = "Predicting..."
                prediction = predictor.predict(result.frames, max_frames=max_frames)
                status = f"Prediction: {prediction.label.upper()} ({prediction.confidence:.1%})"
                print(status)
    finally:
        camera.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VisionSpeak CTC lip worker.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--countdown-seconds", type=float, default=2.0)
    parser.add_argument("--capture-seconds", type=float, default=3.0)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--min-detection-rate", type=float, default=0.75)
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Lip checkpoint not found: {args.checkpoint}")
        predictor = LipPredictor(args.checkpoint, args.device)
        action = run_lip_webcam(
            predictor,
            camera_index=args.camera_index,
            countdown_seconds=args.countdown_seconds,
            capture_seconds=args.capture_seconds,
            min_frames=args.min_frames,
            min_detection_rate=args.min_detection_rate,
            max_frames=args.max_frames,
        )
        return EXIT_SWITCH if action == "switch" else EXIT_QUIT
    except Exception as error:
        print(f"Lip worker error: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
