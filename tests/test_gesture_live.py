from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from inference.gesture_worker import run_live


class _Capture:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def read(self):
        return True, self.frame.copy()


class GestureLiveTests(unittest.TestCase):
    def test_live_keys_return_actions_and_close_camera(self) -> None:
        for key, expected in ((ord("m"), "switch"), (ord("q"), "quit")):
            with self.subTest(key=key):
                detector = Mock()
                detector.camera = _Capture(np.zeros((40, 60, 3), dtype=np.uint8))
                detector.window_name = "test"
                detector.open_camera = Mock(return_value=True)
                detector.close_camera = Mock()
                detector.predict_frame = Mock(
                    return_value=("yes", 0.9, np.zeros((40, 60, 3), dtype=np.uint8))
                )

                with (
                    patch("inference.gesture_worker.cv2.imshow"),
                    patch("inference.gesture_worker.cv2.waitKey", return_value=key),
                    patch("inference.gesture_worker.cv2.putText"),
                ):
                    action = run_live(detector)

                self.assertEqual(action, expected)
                detector.close_camera.assert_called_once_with()

    def test_camera_open_failure_is_cleaned_up(self) -> None:
        detector = Mock()
        detector.open_camera = Mock(return_value=False)
        detector.close_camera = Mock()

        self.assertEqual(run_live(detector), "error")
        detector.close_camera.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
