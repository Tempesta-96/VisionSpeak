from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

sys.modules["ultralytics"] = SimpleNamespace(YOLO=object)
from gesture_detector import GestureEvaluator
sys.modules.pop("ultralytics", None)


class FakeBoxes:
    def __init__(self):
        self.conf = np.array([0.91], dtype=np.float32)
        self.cls = np.array([0], dtype=np.int32)

    def __len__(self):
        return 1


class GestureEvaluatorTests(unittest.TestCase):
    def test_evaluate_accepts_device_argument(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "images"
            label_dir = root / "labels"
            image_dir.mkdir()
            label_dir.mkdir()

            image_path = image_dir / "sample.jpg"
            image_path.write_bytes(b"fake-jpg")
            (label_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            evaluator = GestureEvaluator.__new__(GestureEvaluator)
            evaluator.model = SimpleNamespace(
                names={0: "hello"},
                predict=lambda **kwargs: [SimpleNamespace(boxes=FakeBoxes())],
            )
            evaluator.class_names = evaluator.model.names
            evaluator.test_images = image_dir
            evaluator.test_labels = label_dir
            evaluator.output_dir = root / "runs" / "gesture_test"
            evaluator.output_dir.mkdir(parents=True, exist_ok=True)

            metrics = evaluator.evaluate(confidence=0.25, device="cpu")

            self.assertEqual(metrics["total_images"], 1)
            self.assertEqual(metrics["correct"], 1)
            self.assertEqual(metrics["wrong"], 0)
            self.assertEqual(metrics["confusion_matrix"], [[1]])
            self.assertAlmostEqual(metrics["confusion_matrix_accuracy"], 1.0)
            self.assertGreaterEqual(metrics["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
