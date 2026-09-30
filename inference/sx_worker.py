"""SX webcam process; runs in the shared lip/MediaPipe environment."""
import argparse
from collections import deque
from contextlib import ExitStack
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from inference.gesture_live import run_gesture_live
from inference.protocol import EXIT_ERROR, EXIT_QUIT, EXIT_SWITCH, EXIT_BACKEND_SWITCH
from models.SX.features import extract_rgb
from models.SX.inference import choose_device, load_sx_checkpoint


class FrameWindowBuffer:
    """Rolling window of per-frame landmark vectors.

    A windowed transformer needs consecutive frames, but the webcam loop
    classifies one frame at a time. This buffer keeps the most recent
    ``window_size`` vectors so each frame is classified in its temporal
    context.

    Stale context is worse than no context: if the hand disappears, the
    remaining vectors describe a hand that is no longer there, so
    :meth:`clear` empties the buffer on every no-hand frame. Until the window
    fills, the newest real frame is repeated so a prediction is still possible
    instead of reporting nothing for the first ``window_size`` frames.
    """

    def __init__(self, window_size: int, feature_dim: int) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if feature_dim < 1:
            raise ValueError("feature_dim must be at least 1.")
        self.window_size = window_size
        self.feature_dim = feature_dim
        self._frames: deque = deque(maxlen=window_size)

    def __len__(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def append(self, vector) -> None:
        """Add one frame, rejecting a vector of the wrong width."""
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        if array.shape[0] != self.feature_dim:
            raise ValueError(
                f"expected {self.feature_dim} features per frame, got {array.shape[0]}."
            )
        self._frames.append(array)

    def is_full(self) -> bool:
        return len(self._frames) == self.window_size

    def as_tensor(self) -> torch.Tensor | None:
        """Return the window shaped ``(1, window_size, feature_dim)``.

        Returns ``None`` when nothing has been observed yet.
        """
        if not self._frames:
            return None
        frames = list(self._frames)
        if len(frames) < self.window_size:
            # Hold the newest real frame for the leading positions.
            frames = [frames[-1]] * (self.window_size - len(frames)) + frames
        return torch.from_numpy(np.stack(frames, axis=0)).unsqueeze(0)


class SXDetector:
    def __init__(self, checkpoint, camera_index=0, confidence=.5, device="auto"):
        self.device = choose_device(device)
        self.model, labels, self.config = load_sx_checkpoint(checkpoint, self.device)
        self.labels = {v:k for k,v in labels.items()}
        self.camera_index = camera_index
        self.confidence = confidence
        self.window_name = "VisionSpeak - SX"
        self.camera = None
        self.resources = ExitStack()
        self.buffer = FrameWindowBuffer(self.config.window_size, self.config.input_size)

    def open_camera(self):
        from processing.mediapipe_compat import mp
        try:
            self.hands = self.resources.enter_context(mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=2,
                min_detection_confidence=.5, min_tracking_confidence=.5))
            self.pose = self.resources.enter_context(mp.solutions.pose.Pose(
                static_image_mode=True, min_detection_confidence=.5,
                min_tracking_confidence=.5)) if self.config.use_body_pose else None
            self.camera = cv2.VideoCapture(self.camera_index)
            return self.camera.isOpened()
        except Exception:
            self.close_camera()
            raise

    def close_camera(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.resources.close()
        self.buffer.clear()
        cv2.destroyAllWindows()

    @torch.inference_mode()
    def predict_frame(self, frame):
        vector, hands = extract_rgb(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            two_hands=self.config.two_hands, use_body_pose=self.config.use_body_pose,
            hands=self.hands, pose=self.pose)
        label, confidence = "No hand detected", 0.
        if hands:
            self.buffer.append(vector)
            inputs = self.buffer.as_tensor().to(self.device)
            probability, index = self.model(inputs).softmax(dim=1).max(dim=1)
            confidence = probability.item()
            label = self.labels[index.item()] if confidence >= self.confidence else "Low confidence"
        else:
            # Drop stale context so the next prediction is not built on frames
            # from a hand that has already left the frame.
            self.buffer.clear()
        def render():
            preview = frame.copy()
            for hand in hands or []:
                xs = [p.x * frame.shape[1] for p in hand.landmark]
                ys = [p.y * frame.shape[0] for p in hand.landmark]
                cv2.rectangle(preview, (max(0,int(min(xs))),max(0,int(min(ys)))),
                    (min(frame.shape[1]-1,int(max(xs))),min(frame.shape[0]-1,int(max(ys)))),
                    (0,255,255),2)
            return preview
        return label, confidence, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--confidence', type=float, default=.5)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        from processing.mediapipe_compat import mp
        if not hasattr(mp, 'solutions') or not all(hasattr(mp.solutions, name) for name in ('hands', 'pose')):
            raise RuntimeError('Install requirements.txt in the shared environment; SX needs MediaPipe 0.10.21.')
        detector = SXDetector(args.checkpoint,args.camera_index,args.confidence,args.device)
        if args.check_only:
            return EXIT_QUIT
        variant = 'hands + body' if detector.config.use_body_pose else 'hands only'
        action = run_gesture_live(detector,title=f'SX | {variant} | {args.checkpoint.name}')
        return {'quit':EXIT_QUIT,'switch':EXIT_SWITCH,'backend_switch':EXIT_BACKEND_SWITCH}.get(action,EXIT_ERROR)
    except Exception as error:
        print(f'SX worker error: {error}',file=sys.stderr)
        return EXIT_ERROR


if __name__ == '__main__':
    raise SystemExit(main())
