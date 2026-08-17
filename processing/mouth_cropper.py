"""MediaPipe-based mouth cropping for full video frames."""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MouthCropper:
    """Crop and resize the mouth region of a face to a fixed size.

    Uses MediaPipe Face Landmarker to locate the lips, builds a padded
    bounding box around the lip landmarks, then resizes the crop to
    ``target_size`` (width, height). Falls back to the last known box, then a
    center crop, when no face is detected.
    """

    _LIP_LANDMARK_IDS = (
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    )

    def __init__(
        self,
        target_size: tuple[int, int] = (100, 50),
        model_path: str | Path | None = None,
    ) -> None:
        self.target_size = target_size  # (width, height)
        self._model_path = (
            Path(model_path)
            if model_path is not None
            else Path(__file__).resolve().parent / "face_landmarker.task"
        )
        self._landmarker: vision.FaceLandmarker | None = None
        self._last_crop_box: tuple[int, int, int, int] | None = None

    def __enter__(self) -> "MouthCropper":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def open(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Face landmarker model not found at: {self._model_path}"
            )
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(self._model_path)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            self._last_crop_box = None

    def _detect_landmarks(self, frame: np.ndarray, timestamp_ms: int):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    def _center_crop(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        side = min(height, width)
        x1 = (width - side) // 2
        y1 = (height - side) // 2
        return frame[y1 : y1 + side, x1 : x1 + side]

    def crop(self, frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """Return the mouth region of ``frame`` resized to ``target_size``."""
        if self._landmarker is None:
            raise RuntimeError(
                "MouthCropper is not open; call open() or use a context manager."
            )

        crop_box = self._last_crop_box
        landmarks = self._detect_landmarks(frame, timestamp_ms)
        if landmarks is not None:
            height, width = frame.shape[:2]
            lip_points = np.array(
                [
                    (
                        int(landmarks[idx].x * width),
                        int(landmarks[idx].y * height),
                    )
                    for idx in self._LIP_LANDMARK_IDS
                ],
                dtype=np.int32,
            )
            x_min = int(lip_points[:, 0].min())
            x_max = int(lip_points[:, 0].max())
            y_min = int(lip_points[:, 1].min())
            y_max = int(lip_points[:, 1].max())

            mouth_width = max(x_max - x_min, 1)
            mouth_height = max(y_max - y_min, 1)
            pad_x = max(int(mouth_width * 0.25), 12)
            pad_y = max(int(mouth_height * 0.6), 12)

            x1 = max(x_min - pad_x, 0)
            y1 = max(y_min - pad_y, 0)
            x2 = min(x_max + pad_x, width)
            y2 = min(y_max + pad_y, height)

            if x2 > x1 and y2 > y1:
                crop_box = (x1, y1, x2, y2)
                self._last_crop_box = crop_box

        if crop_box is None:
            cropped = self._center_crop(frame)
        else:
            x1, y1, x2, y2 = crop_box
            cropped = frame[y1:y2, x1:x2]

        return cv2.resize(cropped, self.target_size)
