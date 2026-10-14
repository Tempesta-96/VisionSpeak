"""Tests for ``evaluate_epoch_sweep.py``.

These cover the grid bookkeeping that has to be right for the 3x4 table to mean
anything: that every architecture/epoch cell resolves to a checkpoint, that a
checkpoint contradicting its cell is refused rather than tabulated, and that the
rendered table and written artefacts contain the numbers that were measured.

Nothing here needs MediaPipe, a GPU, or the real checkpoints: the guards are
exercised against tiny checkpoints written into a temp directory, and the
rendering against synthetic summaries. The mouth-cropping and prediction paths
belong to ``compare_lip_models`` and are covered by
``tests/test_compare_lip_models.py``.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

import compare_lip_models as comparison
import evaluate_epoch_sweep as sweep

ONE_ARCHITECTURE = (("ctc_bigru", "CTC + BiGRU"),)
TWO_ARCHITECTURES = (
    ("ctc_bigru", "CTC + BiGRU"),
    ("word_transformer_attention", "Word + Transformer"),
)


def write_checkpoint(directory: Path, name: str, **metadata) -> Path:
    """Write a minimal checkpoint carrying only the metadata the guards read."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": 50, "test_speaker": "speaker_test_samples", "label_to_index": {"no": 0}}, path)
    if metadata:
        existing = torch.load(path, map_location="cpu", weights_only=True)
        existing.update(metadata)
        torch.save(existing, path)
    return path


def synthetic_summaries(epochs, architectures):
    """Summaries where epoch and column index together set the accuracy."""
    summaries = {}
    for epoch_index, epoch in enumerate(epochs):
        for column_index, (key, _) in enumerate(architectures):
            correct = 10 * (epoch_index + 1) + column_index
            summaries[(epoch, key)] = {
                "accuracy": correct / 160,
                "correct": correct,
                "predicted": 160,
                "accuracy_ci95": [0.1, 0.3],
                "status_counts": {"predicted": 160},
            }
    return summaries


class BuildGridTests(unittest.TestCase):
    def test_full_grid_is_three_epoch_rows_of_four_architectures(self) -> None:
        grid = sweep.build_grid(sweep.DEFAULT_CHECKPOINT_DIR)
        self.assertEqual(len(grid), len(sweep.EPOCH_BUDGETS))
        for row in grid:
            self.assertEqual(len(row), len(sweep.ARCHITECTURES))
        # Rows are epochs, columns are architectures, in the declared order.
        self.assertEqual([spec.tag for spec in grid[0]][:2], ["ctc_bigru_ep50", "ctc_transformer_ep50"])
        self.assertEqual([spec.tag for spec in grid[-1]][:2], ["ctc_bigru_ep200", "ctc_transformer_ep200"])

    def test_missing_checkpoints_are_all_named(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(sweep.SweepError) as caught:
                sweep.build_grid(Path(temporary), (50, 100), TWO_ARCHITECTURES)
        message = str(caught.exception)
        # Every missing cell, not just the first one.
        self.assertIn("CTC + BiGRU @ 50 epochs", message)
        self.assertIn("Word + Transformer @ 100 epochs", message)
        self.assertIn("retrain_lip_models.sh", message)

    def test_present_checkpoints_produce_a_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_checkpoint(directory, "ctc_bigru_ep50.pt")
            grid = sweep.build_grid(directory, (50,), ONE_ARCHITECTURE)
        self.assertEqual(len(grid), 1)
        self.assertEqual(grid[0][0].tag, "ctc_bigru_ep50")


class ValidateGridTests(unittest.TestCase):
    def test_matching_checkpoints_raise_no_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_checkpoint(directory, "ctc_bigru_ep50.pt")
            grid = sweep.build_grid(directory, (50,), ONE_ARCHITECTURE)
            warnings = sweep.validate_grid(grid, (50,), ONE_ARCHITECTURE)
        self.assertEqual(warnings, [])

    def test_recorded_epoch_contradicting_its_row_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            # A 200-epoch checkpoint sitting in the 50-epoch row.
            write_checkpoint(directory, "ctc_bigru_ep50.pt", epoch=200)
            grid = sweep.build_grid(directory, (50,), ONE_ARCHITECTURE)
            with self.assertRaises(sweep.SweepError) as caught:
                sweep.validate_grid(grid, (50,), ONE_ARCHITECTURE)
        self.assertIn("records epoch 200", str(caught.exception))

    def test_checkpoint_trained_on_another_speaker_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_checkpoint(directory, "ctc_bigru_ep50.pt", test_speaker="TestID")
            grid = sweep.build_grid(directory, (50,), ONE_ARCHITECTURE)
            with self.assertRaises(sweep.SweepError) as caught:
                sweep.validate_grid(grid, (50,), ONE_ARCHITECTURE, expected_speaker="speaker_test_samples")
        self.assertIn("would not be held out", str(caught.exception))

    def test_speaker_mismatch_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_checkpoint(directory, "ctc_bigru_ep50.pt", test_speaker="TestID")
            grid = sweep.build_grid(directory, (50,), ONE_ARCHITECTURE)
            warnings = sweep.validate_grid(
                grid,
                (50,),
                ONE_ARCHITECTURE,
                expected_speaker="speaker_test_samples",
                allow_speaker_mismatch=True,
            )
        self.assertEqual(warnings, [])

    def test_checkpoint_without_a_recorded_speaker_warns_instead_of_failing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "ctc_bigru_ep50.pt"
            torch.save({"epoch": 50}, path)
            grid = sweep.build_grid(directory, (50,), ONE_ARCHITECTURE)
            warnings = sweep.validate_grid(grid, (50,), ONE_ARCHITECTURE)
        self.assertEqual(len(warnings), 1)
        self.assertIn("does not record test_speaker", warnings[0])


class RenderingTests(unittest.TestCase):
    def test_accuracy_grid_is_the_requested_three_by_four_percentages(self) -> None:
        summaries = synthetic_summaries(sweep.EPOCH_BUDGETS, sweep.ARCHITECTURES)
        rows = sweep.render_grid(
            lambda epoch, key: sweep.format_percent(summaries[(epoch, key)]["accuracy"])
        )
        self.assertEqual(rows[0], ["epoch", "CTC + BiGRU", "CTC + Transformer", "Word + BiGRU", "Word + Transformer"])
        self.assertEqual(rows[1], ["50", "6.25%", "6.88%", "7.50%", "8.12%"])
        self.assertEqual(len(rows), len(sweep.EPOCH_BUDGETS) + 1)

    def test_count_grid_uses_the_predicted_denominator(self) -> None:
        summaries = synthetic_summaries((50,), ONE_ARCHITECTURE)
        summaries[(50, "ctc_bigru")].update({"correct": 20, "predicted": 150})
        rows = sweep.render_grid(
            lambda epoch, key: f"{summaries[(epoch, key)]['correct']}/{summaries[(epoch, key)]['predicted']}",
            (50,),
            ONE_ARCHITECTURE,
        )
        self.assertEqual(rows[1], ["50", "20/150"])

    def test_plain_table_aligns_and_has_a_separator(self) -> None:
        rendered = sweep.format_table([["epoch", "CTC + BiGRU"], ["50", "12.50%"]])
        lines = rendered.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(set(lines[1]) <= {"-", " "})
        self.assertIn("12.50%", lines[2])

    def test_markdown_table_has_a_header_separator(self) -> None:
        rendered = sweep.format_markdown([["epoch", "a"], ["50", "1.00%"]])
        self.assertIn("| epoch | a |", rendered)
        self.assertIn("| --- | --- |", rendered)
        # The first data row must survive: it is easy to mistake for a separator.
        lines = rendered.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[2], "| 50 | 1.00% |")


class ArtefactTests(unittest.TestCase):
    def test_written_artefacts_contain_the_measured_numbers(self) -> None:
        summaries = synthetic_summaries((50, 100), TWO_ARCHITECTURES)
        per_clip = [
            {
                "architecture": "CTC + BiGRU", "epoch": 50, "sample_id": "clip_1", "label": "no",
                "predicted": "no", "confidence": "0.900000", "correct": True, "status": "predicted",
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            markdown, scores, predictions = sweep.write_artifacts(
                summaries, Path(temporary), "sweep", epochs=(50, 100), architectures=TWO_ARCHITECTURES,
                per_clip=per_clip,
            )
            markdown_text = markdown.read_text()
            with scores.open() as handle:
                score_rows = list(csv.DictReader(handle))
            with predictions.open() as handle:
                clip_rows = list(csv.DictReader(handle))

        self.assertIn("6.25%", markdown_text)
        self.assertIn("| epoch | CTC + BiGRU | Word + Transformer |", markdown_text)
        self.assertEqual(len(score_rows), 4)  # 2 epochs x 2 architectures
        self.assertEqual(score_rows[0]["architecture"], "CTC + BiGRU")
        self.assertEqual(score_rows[0]["accuracy"], f"{10 / 160:.6f}")
        self.assertEqual(len(clip_rows), 1)
        self.assertEqual(clip_rows[0]["sample_id"], "clip_1")


class ArgumentTests(unittest.TestCase):
    def test_epochs_parse_from_a_comma_separated_list(self) -> None:
        self.assertEqual(sweep.parse_epochs("50, 100 ,200"), (50, 100, 200))

    def test_empty_epoch_list_is_rejected(self) -> None:
        with self.assertRaises(Exception):
            sweep.parse_epochs(" , ")


class CacheKeyTests(unittest.TestCase):
    def test_cache_key_matches_compare_lip_models(self) -> None:
        """The reported cache key must be the one the frames were really stored under."""
        for min_frames, rate in ((30, 0.75), (45, 0.9)):
            expected = comparison.cache_path(
                Path(".cache/lip_mp4_frames"), "clip_1", min_frames=min_frames, min_detection_rate=rate
            ).name
            actual = sweep.CACHE_KEY_TEMPLATE.format(
                min_frames=min_frames, min_detection_rate=rate
            ).format(sample_id="clip_1")
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
