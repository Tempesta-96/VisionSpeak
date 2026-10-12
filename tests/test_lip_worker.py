from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from inference.lip_worker import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    WORDS,
    LipPredictor,
    clip_quality_message,
    prepare_clip,
    score_vocabulary,
    run_lip_webcam,
)
from models.valente.model import LipNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CTC_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "valente_custom_words_lipnet_ctc_0.852_acc.pt"
)


class _Camera:
    def __init__(self) -> None:
        self.released = False

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        self.released = True


class _CropperContext:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


class LipWorkerTests(unittest.TestCase):
    def test_prepare_clip_preserves_input_and_pads_model_shape(self) -> None:
        frames = np.arange(40 * FRAME_HEIGHT * FRAME_WIDTH, dtype=np.uint8).reshape(
            40, FRAME_HEIGHT, FRAME_WIDTH
        )
        original = frames.copy()

        tensor, lengths = prepare_clip(frames, max_frames=90)

        self.assertEqual(tensor.shape, (1, 90, 1, FRAME_HEIGHT, FRAME_WIDTH))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(lengths.tolist(), [40])
        self.assertTrue(torch.equal(tensor[:, 40:], torch.zeros_like(tensor[:, 40:])))
        np.testing.assert_array_equal(frames, original)

    def test_prepare_clip_resamples_long_input(self) -> None:
        frames = np.zeros((120, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        tensor, lengths = prepare_clip(frames, max_frames=90)
        self.assertEqual(tensor.shape[1], 90)
        self.assertEqual(lengths.tolist(), [90])

    def test_ctc_scoring_prefers_the_matching_allowed_word(self) -> None:
        logits = torch.full((5, 1, 3), -8.0)
        for timestep, token in enumerate((1, 1, 0, 2, 2)):
            logits[timestep, 0, token] = 8.0
        scores = score_vocabulary(
            torch.log_softmax(logits, dim=-1),
            torch.tensor([5]),
            ("hi", "ih"),
            {"h": 1, "i": 2},
        )

        self.assertGreater(scores["hi"], scores["ih"])
        self.assertAlmostEqual(sum(scores.values()), 1.0, places=6)

    def test_clip_quality_uses_recorder_thresholds(self) -> None:
        self.assertIsNone(
            clip_quality_message(30, 0.75, min_frames=30, min_detection_rate=0.75)
        )
        self.assertIn(
            "at least 30",
            clip_quality_message(29, 1.0, min_frames=30, min_detection_rate=0.75),
        )
        self.assertIn(
            "75%",
            clip_quality_message(30, 0.5, min_frames=30, min_detection_rate=0.75),
        )

    def test_webcam_quit_releases_camera_and_windows(self) -> None:
        camera = _Camera()
        full_frame = np.zeros((80, 160, 3), dtype=np.uint8)
        mouth_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        with (
            patch("inference.lip_worker.cv2.VideoCapture", return_value=camera),
            patch("inference.lip_worker.cv2.imshow"),
            patch("inference.lip_worker.cv2.waitKey", return_value=ord("q")),
            patch("inference.lip_worker.cv2.destroyAllWindows") as destroy_windows,
            patch("inference.lip_worker.MouthCropper", return_value=_CropperContext()),
            patch(
                "inference.lip_worker._read_mouth_frame",
                return_value=(full_frame, mouth_frame, None, False),
            ),
        ):
            action = run_lip_webcam(
                Mock(),
                camera_index=0,
                countdown_seconds=2.0,
                capture_seconds=3.0,
                min_frames=30,
                min_detection_rate=0.75,
                max_frames=90,
            )

        self.assertEqual(action, "quit")
        self.assertTrue(camera.released)
        destroy_windows.assert_called_once_with()


class LipPredictorArchitectureTests(unittest.TestCase):
    """The worker must serve both temporal cores from one code path."""

    def _frames(self) -> np.ndarray:
        generator = np.random.default_rng(0)
        return (generator.random((60, FRAME_HEIGHT, FRAME_WIDTH)) * 255).astype(np.uint8)

    def _write_transformer_checkpoint(self, path: Path, char_to_index: dict) -> None:
        model = LipNet(num_classes=len(char_to_index) + 1, architecture="transformer")
        with torch.inference_mode():
            model(torch.zeros(1, 10, 1, FRAME_HEIGHT, FRAME_WIDTH), torch.tensor([10]))
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "char_to_index": char_to_index,
                "architecture": "transformer",
                "hidden_size": 256,
                "transformer_layers": 3,
                "transformer_heads": 4,
                "transformer_feedforward": 512,
            },
            path,
        )

    def test_legacy_checkpoint_still_loads_as_bigru(self) -> None:
        predictor = LipPredictor(LEGACY_CTC_CHECKPOINT, "cpu")
        self.assertEqual(predictor.architecture, "bigru")
        prediction = predictor.predict(self._frames())
        self.assertIn(prediction.label, WORDS)
        self.assertAlmostEqual(sum(prediction.scores.values()), 1.0, places=5)

    def test_transformer_checkpoint_selected_from_metadata(self) -> None:
        legacy = LipPredictor(LEGACY_CTC_CHECKPOINT, "cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transformer_ctc.pt"
            self._write_transformer_checkpoint(path, legacy.char_to_index)
            predictor = LipPredictor(path, "cpu")

        self.assertEqual(predictor.architecture, "transformer")
        prediction = predictor.predict(self._frames())
        self.assertIn(prediction.label, WORDS)
        self.assertAlmostEqual(sum(prediction.scores.values()), 1.0, places=5)

    def test_architecture_mismatch_reports_the_recorded_name(self) -> None:
        legacy = LipPredictor(LEGACY_CTC_CHECKPOINT, "cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "misdeclared.pt"
            self._write_transformer_checkpoint(path, legacy.char_to_index)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            # Weights are a transformer's but the metadata claims BiGRU.
            checkpoint["architecture"] = "bigru"
            torch.save(checkpoint, path)

            with self.assertRaisesRegex(ValueError, "bigru"):
                LipPredictor(path, "cpu")


if __name__ == "__main__":
    unittest.main()
