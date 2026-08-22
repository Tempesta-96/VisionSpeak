"""Tests for the isolated-word prediction module."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.LBY import predict


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "checkpoints" / "best.pt"
PROCESSED_DIR = ROOT / "data" / "lips-dataset" / "processedData"
SAMPLE_ARCHIVE = next(PROCESSED_DIR.glob("*.npz"), None)


class WordPredictorTests(unittest.TestCase):
    def test_load_model_restores_checkpoint_in_evaluation_mode(self) -> None:
        model, index_to_word = predict.load_model(CHECKPOINT_PATH, torch.device("cpu"))

        self.assertFalse(model.training)
        self.assertEqual(len(index_to_word), 51)
        self.assertEqual(len(set(index_to_word.values())), 51)

    def test_prepare_frames_matches_model_input_contract(self) -> None:
        frames = np.full((4, 50, 100), 128, dtype=np.uint8)

        tensor = predict.prepare_frames(frames)

        self.assertEqual(tuple(tensor.shape), (1, 4, 1, 50, 100))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    @unittest.skipUnless(
        SAMPLE_ARCHIVE is not None,
        "Processed GRID data is required for this inference smoke test.",
    )
    def test_predict_frames_returns_checkpoint_vocabulary_word(self) -> None:
        model, index_to_word = predict.load_model(CHECKPOINT_PATH, torch.device("cpu"))
        with np.load(SAMPLE_ARCHIVE, allow_pickle=False) as archive:
            length = int(archive["segment_lengths"][0])
            frames = archive["word_segments"][0, :length]

        word = predict.predict_frames(frames, model, index_to_word, torch.device("cpu"))

        self.assertIn(word, index_to_word.values())

    def test_main_prints_only_predicted_word(self) -> None:
        output = StringIO()
        with (
            patch.object(sys, "argv", ["predict", "--video-path", "word.mp4"]),
            patch.object(predict, "predict_word", return_value="blue"),
            redirect_stdout(output),
        ):
            predict.main()

        self.assertEqual(output.getvalue(), "blue\n")

    def test_missing_video_path_is_clear(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Video file not found"):
            predict.predict_word("does-not-exist.mp4")

    def test_corrupt_checkpoint_is_clear(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "corrupt.pt"
            checkpoint_path.write_text("not a torch checkpoint")

            with self.assertRaisesRegex(ValueError, "Could not load"):
                predict.load_model(checkpoint_path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
