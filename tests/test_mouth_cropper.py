from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from processing.mouth_cropper import MouthCropper


class MouthCropperMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.full((80, 160, 3), 100, dtype=np.uint8)
        self.landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(406)]
        self.landmarks[61] = SimpleNamespace(x=0.4, y=0.45)
        self.landmarks[291] = SimpleNamespace(x=0.6, y=0.55)

    def test_metadata_distinguishes_fresh_and_cached_boxes(self) -> None:
        cropper = MouthCropper()
        cropper._landmarker = object()
        with patch.object(
            cropper,
            "_detect_landmarks",
            side_effect=(self.landmarks, None),
        ):
            first_crop, first_box, first_fresh = cropper.crop_with_metadata(self.frame, 0)
            second_crop, second_box, second_fresh = cropper.crop_with_metadata(self.frame, 1)

        self.assertEqual(first_crop.shape, (50, 100, 3))
        self.assertEqual(second_crop.shape, (50, 100, 3))
        self.assertEqual(first_box, second_box)
        self.assertTrue(first_fresh)
        self.assertFalse(second_fresh)

    def test_reset_tracking_removes_cached_box(self) -> None:
        cropper = MouthCropper()
        cropper._last_crop_box = (1, 2, 3, 4)
        cropper.reset_tracking()
        self.assertIsNone(cropper._last_crop_box)


if __name__ == "__main__":
    unittest.main()
