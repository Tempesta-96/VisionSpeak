"""Tests for VideoLoader webcam preview behaviour without a real camera."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from processing.video_loader import VideoLoader


class _LandmarkerContext:
    def __enter__(self) -> object:
        return object()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


class _Capture:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        return True, self.frame.copy()

    def release(self) -> None:
        self.released = True


class VideoLoaderPreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.full((80, 160, 3), 100, dtype=np.uint8)
        self.mouth_crop = np.full((32, 56, 3), 200, dtype=np.uint8)
        self.crop_box = (52, 24, 108, 56)

    def _load_with_mocked_webcam(
        self,
        *,
        show_full_frame: bool,
        crop_box: tuple[int, int, int, int] | None,
    ):
        capture = _Capture(self.frame)
        loader = VideoLoader()
        with (
            patch.object(
                loader,
                "_create_face_landmarker",
                return_value=_LandmarkerContext(),
            ),
            patch.object(
                loader,
                "_crop_mouth_area",
                return_value=(self.mouth_crop, crop_box),
            ),
            patch("processing.video_loader.cv2.VideoCapture", return_value=capture),
            patch("processing.video_loader.cv2.waitKey", return_value=ord("q")),
            patch("processing.video_loader.cv2.imshow") as imshow,
            patch("processing.video_loader.cv2.rectangle") as rectangle,
            patch("processing.video_loader.cv2.putText") as put_text,
        ):
            video_data = loader.load("webcam", show_full_frame=show_full_frame)
        return video_data, capture, imshow, rectangle, put_text

    def test_crop_returns_padded_lip_box(self) -> None:
        loader = VideoLoader()
        landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(406)]
        landmarks[61] = SimpleNamespace(x=0.4, y=0.45)
        landmarks[291] = SimpleNamespace(x=0.6, y=0.55)

        with patch.object(loader, "_detect_landmarks", return_value=landmarks):
            crop, crop_box = loader._crop_mouth_area(self.frame, object(), 0)

        self.assertEqual(crop_box, self.crop_box)
        self.assertEqual(crop.shape, self.mouth_crop.shape)

    def test_full_frame_preview_draws_the_crop_box_and_stores_resized_crops(self) -> None:
        video_data, capture, imshow, rectangle, _ = self._load_with_mocked_webcam(
            show_full_frame=True,
            crop_box=self.crop_box,
        )

        displayed_frame = imshow.call_args.args[1]
        self.assertEqual(displayed_frame.shape, self.frame.shape)
        self.assertFalse(np.shares_memory(displayed_frame, self.frame))
        rectangle.assert_called_once_with(
            displayed_frame,
            (52, 24),
            (107, 55),
            (0, 255, 0),
            2,
            16,
        )
        self.assertEqual(video_data["frames"].shape, (1, 50, 100, 3))
        capture.release()
        self.assertTrue(capture.released)

    def test_full_frame_preview_reports_missing_lips_without_a_box(self) -> None:
        _, capture, imshow, rectangle, put_text = self._load_with_mocked_webcam(
            show_full_frame=True,
            crop_box=None,
        )

        self.assertEqual(imshow.call_args.args[1].shape, self.frame.shape)
        rectangle.assert_not_called()
        self.assertEqual(put_text.call_args.args[1], "Lips not detected")
        capture.release()

    def test_default_preview_remains_the_mouth_crop(self) -> None:
        _, capture, imshow, rectangle, put_text = self._load_with_mocked_webcam(
            show_full_frame=False,
            crop_box=self.crop_box,
        )

        np.testing.assert_array_equal(imshow.call_args.args[1], self.mouth_crop)
        rectangle.assert_not_called()
        put_text.assert_not_called()
        capture.release()


if __name__ == "__main__":
    unittest.main()
