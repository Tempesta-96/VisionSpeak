from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class VideoLoader:
    """Minimal video loader for MP4 files and webcam input."""

    # MediaPipe Face Mesh lip landmarks used to build a mouth bounding box.
    _LIP_LANDMARK_IDS = (
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    )

    def __init__(self) -> None:
        self._model_path = Path(__file__).resolve(
        ).parent / "face_landmarker.task"
        self._last_crop_box: tuple[int, int, int, int] | None = None

    def _create_face_landmarker(self, running_mode: vision.RunningMode):
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(self._model_path)),
            running_mode=running_mode,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def _center_crop(self, frame, crop_size: int = 120):
        height, width = frame.shape[:2]
        half = crop_size // 2
        center_x = width // 2
        center_y = height // 2
        x1 = max(center_x - half, 0)
        y1 = max(center_y - half, 0)
        x2 = min(center_x + half, width)
        y2 = min(center_y + half, height)
        return frame[y1:y2, x1:x2]

    def _crop_from_box(self, frame, box: tuple[int, int, int, int]):
        x1, y1, x2, y2 = box
        return frame[y1:y2, x1:x2]

    def _detect_landmarks(self, frame, landmarker: vision.FaceLandmarker, timestamp_ms: int | None = None,):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if timestamp_ms is None:
            result = landmarker.detect(mp_image)
        else:
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    def _crop_mouth_area(
        self,
        frame,
        landmarker: vision.FaceLandmarker,
        timestamp_ms: int | None = None,
    ) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        """Return a mouth crop and the padded box used to produce it.

        A missing detection reuses the last valid box when possible. If no box
        has been detected yet, the method returns a center crop and ``None``
        so callers can distinguish that fallback from a lip detection.
        """
        landmarks = self._detect_landmarks(frame, landmarker, timestamp_ms)
        crop_box = None
        if landmarks is not None:
            height, width = frame.shape[:2]
            lip_points = np.array(
                [
                    (int(landmarks[idx].x * width), int(landmarks[idx].y * height))
                    for idx in self._LIP_LANDMARK_IDS
                ],
                dtype=np.int32,
            )

            x_coords = lip_points[:, 0]
            y_coords = lip_points[:, 1]
            x_min = int(x_coords.min())
            x_max = int(x_coords.max())
            y_min = int(y_coords.min())
            y_max = int(y_coords.max())

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

        crop_box = crop_box or self._last_crop_box
        if crop_box is not None:
            return self._crop_from_box(frame, crop_box), crop_box
        return self._center_crop(frame), None

    @staticmethod
    def _full_frame_preview(
        frame: np.ndarray,
        crop_box: tuple[int, int, int, int] | None,
    ) -> np.ndarray:
        """Draw the mouth-crop box on a full webcam frame for display."""
        preview = frame.copy()
        if crop_box is None:
            cv2.putText(
                preview,
                "Lips not detected",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return preview

        x1, y1, x2, y2 = crop_box
        cv2.rectangle(
            preview,
            (x1, y1),
            (x2 - 1, y2 - 1),
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            "Mouth crop",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return preview

    def load(self, source: str, *, show_full_frame: bool = False) -> dict:
        frames = []
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Face landmarker model not found at: {self._model_path}"
            )

        with self._create_face_landmarker(vision.RunningMode.VIDEO) as landmarker:
            if source == "webcam":
                capture = cv2.VideoCapture(0)
                if not capture.isOpened():
                    raise RuntimeError(
                        "Unable to connect to the default webcam device.")
                frame_index = 0
                while True:
                    success, frame = capture.read()
                    if not success:
                        break

                    mouth_frame, crop_box = self._crop_mouth_area(
                        frame,
                        landmarker,
                        timestamp_ms=frame_index * 33,
                    )
                    frames.append(mouth_frame)
                    preview_frame = (
                        self._full_frame_preview(frame, crop_box)
                        if show_full_frame
                        else mouth_frame
                    )
                    cv2.imshow("VisionSpeak Webcam", preview_frame)
                    frame_index += 1

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    
                frames = [cv2.resize(frame, (100, 50)) for frame in frames]
                frames = np.array(frames, dtype=np.float32)   
                return {
                    "source_type": "webcam",
                    "path": None,
                    "capture": capture,
                    "frames": frames,
                    "loaded": len(frames) > 0,
                }

            video_path = Path(source)
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if fps and fps > 0 else 30.0
            frame_index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                timestamp_ms = int((frame_index / fps) * 1000)
                mouth_frame, _ = self._crop_mouth_area(
                    frame, 
                    landmarker, 
                    timestamp_ms)
                frames.append(mouth_frame)
                frame_index += 1

            frames = [cv2.resize(frame, (100, 50)) for frame in frames]
            frames = np.array(frames, dtype=np.float32)
            capture.release()
            return {
                "source_type": "video_file",
                "path": str(video_path),
                "frames": frames,
                "loaded": len(frames) > 0,
            }
