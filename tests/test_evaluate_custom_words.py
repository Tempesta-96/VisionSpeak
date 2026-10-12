from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

import evaluate_custom_words as ec
from inference.lip_worker import FRAME_HEIGHT, FRAME_WIDTH, TimestampClock, WORDS


class FakePredictor:
    """Stand-in for LipPredictor that records the frames it receives."""

    def __init__(
        self,
        checkpoint=None,
        device_name: str = "auto",
        *,
        label: str = "hello",
        confidence: float = 0.75,
    ) -> None:
        self.label = label
        self.confidence = confidence
        self.received: list[np.ndarray] = []
        remainder = (1.0 - confidence) / max(len(WORDS) - 1, 1)
        self.scores = {
            word: confidence if word == label else remainder for word in WORDS
        }
        self.prediction_mode = "ctc"
        self.architecture = "bigru"
        self.device = "cpu"

    def predict(self, frames, *, max_frames: int = 90):
        self.received.append(np.asarray(frames))
        return SimpleNamespace(
            label=self.label,
            confidence=self.confidence,
            scores=dict(self.scores),
        )


class FakeCropper:
    """Stand-in for MouthCropper with a fixed 50x100 BGR crop."""

    def __init__(self, fresh: bool = True) -> None:
        self.fresh = fresh
        self.calls = 0
        self.resets = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def reset_tracking(self) -> None:
        self.resets += 1

    def crop_with_metadata(self, frame, timestamp_ms: int):
        self.calls += 1
        crop = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 128, dtype=np.uint8)
        return crop, (0, 0, 8, 8), self.fresh


def make_row(**overrides) -> SimpleNamespace:
    """Build a manifest-like row with test-friendly defaults."""
    values = dict(
        sample_id="s1",
        speaker_id="spk_a",
        label="hello",
        repetition=1,
        relative_path="clips/spk_a/hello/s1.npz",
        raw_video_path="raw/spk_a/hello/s1.mp4",
        frame_count=40,
        fps=30.0,
        duration_seconds=1.3,
        detection_rate=1.0,
        captured_at="2026-01-01T00:00:00+00:00",
        clip_path="/nonexistent/clips/spk_a/hello/s1.npz",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_result(**overrides) -> ec.ClipResult:
    """Build a ClipResult with test-friendly defaults."""
    values = dict(
        sample_id="s1",
        speaker_id="spk_a",
        label="hello",
        repetition=1,
        predicted="hello",
        confidence=0.9,
        correct=True,
        status="predicted",
        note="",
    )
    values.update(overrides)
    return ec.ClipResult(**values)


class SelectRowsTests(unittest.TestCase):
    def test_no_filter_returns_everything(self) -> None:
        manifest = pd.DataFrame(
            {"sample_id": ["a", "b"], "speaker_id": ["spk_a", "spk_b"], "label": ["hello", "hi"]}
        )
        rows, unknown = ec.select_rows(manifest, None)
        self.assertEqual(len(rows), 2)
        self.assertEqual(unknown, [])

    def test_filter_keeps_selected_speakers_and_flags_unknown(self) -> None:
        manifest = pd.DataFrame(
            {"sample_id": ["a", "b", "c"], "speaker_id": ["spk_a", "spk_a", "spk_b"], "label": ["hello"] * 3}
        )
        rows, unknown = ec.select_rows(manifest, ["spk_a", "ghost"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(unknown, ["ghost"])


class NpzSourceTests(unittest.TestCase):
    def test_prediction_passes_frames_through_and_scores_the_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frames = np.full((6, FRAME_HEIGHT, FRAME_WIDTH), 200, dtype=np.uint8)
            archive_path = Path(tmp) / "s1.npz"
            np.savez_compressed(archive_path, frames=frames)

            predictor = FakePredictor(label="hello", confidence=0.8)
            result = ec.predict_npz_result(
                make_row(label="hello", clip_path=str(archive_path)),
                predictor,
                max_frames=90,
            )

        self.assertEqual(result.status, "predicted")
        self.assertEqual(result.predicted, "hello")
        self.assertTrue(result.correct)
        self.assertEqual(result.confidence, 0.8)
        self.assertEqual(len(predictor.received), 1)
        self.assertTrue(np.array_equal(predictor.received[0], frames))

    def test_wrong_prediction_is_marked_incorrect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "s1.npz"
            np.savez_compressed(
                archive_path, frames=np.zeros((6, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
            )
            result = ec.predict_npz_result(
                make_row(label="hello", clip_path=str(archive_path)),
                FakePredictor(label="bye"),
                max_frames=90,
            )
        self.assertEqual(result.status, "predicted")
        self.assertFalse(result.correct)

    def test_missing_archive_is_reported_without_predicting(self) -> None:
        predictor = FakePredictor()
        result = ec.predict_npz_result(make_row(), predictor, max_frames=90)
        self.assertEqual(result.status, "missing-file")
        self.assertIn("missing archive", result.note)
        self.assertEqual(predictor.received, [])


class Mp4SourceTests(unittest.TestCase):
    def _write_video(self, directory: Path, frame_count: int = 20) -> Path:
        video_path = directory / "raw" / "spk_a" / "hello" / "s1.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48)
        )
        self.assertTrue(writer.isOpened())
        for _ in range(frame_count):
            writer.write(np.full((48, 64, 3), 100, dtype=np.uint8))
        writer.release()
        return video_path

    def test_prediction_decodes_video_through_the_cropper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = self._write_video(Path(tmp))
            predictor = FakePredictor(label="hello")
            cropper = FakeCropper(fresh=True)
            result = ec.predict_mp4_result(
                make_row(label="hello", raw_video_path="raw/spk_a/hello/s1.mp4"),
                cropper,
                TimestampClock(),
                predictor,
                data_dir=Path(tmp),
                max_frames=90,
                min_frames=5,
                min_detection_rate=0.5,
            )

        self.assertEqual(result.status, "predicted")
        self.assertTrue(result.correct)
        self.assertEqual(cropper.calls, 20)
        self.assertEqual(cropper.resets, 1)
        self.assertEqual(len(predictor.received), 1)
        self.assertEqual(predictor.received[0].shape, (20, FRAME_HEIGHT, FRAME_WIDTH))

    def test_short_clip_is_rejected_by_the_quality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = self._write_video(Path(tmp), frame_count=20)
            predictor = FakePredictor()
            result = ec.predict_mp4_result(
                make_row(raw_video_path="raw/spk_a/hello/s1.mp4"),
                FakeCropper(),
                TimestampClock(),
                predictor,
                data_dir=Path(tmp),
                max_frames=90,
                min_frames=100,
                min_detection_rate=0.5,
            )

        self.assertEqual(result.status, "rejected")
        self.assertIn("at least 100", result.note)
        self.assertEqual(predictor.received, [])

    def test_shared_clock_keeps_timestamps_monotonic_across_videos(self) -> None:
        class RecordingCropper(FakeCropper):
            def __init__(self) -> None:
                super().__init__()
                self.timestamps: list[int] = []

            def crop_with_metadata(self, frame, timestamp_ms: int):
                self.timestamps.append(timestamp_ms)
                return super().crop_with_metadata(frame, timestamp_ms)

        with tempfile.TemporaryDirectory() as tmp:
            first_video = self._write_video(Path(tmp) / "one")
            second_video = self._write_video(Path(tmp) / "two")
            cropper = RecordingCropper()
            clock = TimestampClock()  # One clock for the whole run, like the live session.
            for video in (first_video, second_video):
                frames, _detection_rate, message = ec.extract_mp4_mouth_frames(
                    video, cropper, clock, min_frames=5, min_detection_rate=0.5
                )
                self.assertIsNotNone(frames)
                self.assertEqual(message, "")

        self.assertEqual(len(cropper.timestamps), 40)
        self.assertTrue(
            all(later > earlier for earlier, later in zip(cropper.timestamps, cropper.timestamps[1:])),
            "timestamps must strictly increase across videos sharing one landmarker",
        )

    def test_row_without_raw_video_is_reported(self) -> None:
        result = ec.predict_mp4_result(
            make_row(raw_video_path=float("nan")),
            FakeCropper(),
            TimestampClock(),
            FakePredictor(),
            data_dir=Path("."),
            max_frames=90,
            min_frames=5,
            min_detection_rate=0.5,
        )
        self.assertEqual(result.status, "missing-file")
        self.assertIn("no raw video", result.note)

    def test_missing_video_file_is_reported(self) -> None:
        result = ec.predict_mp4_result(
            make_row(raw_video_path="raw/spk_a/hello/missing.mp4"),
            FakeCropper(),
            TimestampClock(),
            FakePredictor(),
            data_dir=Path("/nonexistent"),
            max_frames=90,
            min_frames=5,
            min_detection_rate=0.5,
        )
        self.assertEqual(result.status, "missing-file")
        self.assertIn("missing raw video", result.note)


class EvaluateRowsTests(unittest.TestCase):
    def test_per_clip_errors_are_isolated(self) -> None:
        rows = pd.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "speaker_id": ["spk_a"] * 3,
                "label": ["hello"] * 3,
                "repetition": [1, 2, 3],
            }
        )

        def predict_one(row):
            if row.sample_id == "b":
                raise RuntimeError("boom")
            return make_result(sample_id=str(row.sample_id), repetition=int(row.repetition))

        results = ec.evaluate_rows(rows, predict_one)
        self.assertEqual([result.status for result in results], ["predicted", "error", "predicted"])
        self.assertIn("boom", results[1].note)


class ReportTests(unittest.TestCase):
    def test_summarize_counts_accuracy_and_confidence(self) -> None:
        results = [
            make_result(correct=True, confidence=0.9),
            make_result(correct=True, confidence=0.7),
            make_result(correct=False, predicted="bye", confidence=0.6),
            make_result(status="rejected", predicted="", confidence=0.0, correct=False),
        ]
        summary = ec.summarize(results)
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["predicted"], 3)
        self.assertEqual(summary["correct"], 2)
        self.assertAlmostEqual(summary["accuracy"], 2 / 3)
        self.assertAlmostEqual(summary["mean_confidence"], (0.9 + 0.7 + 0.6) / 3)
        self.assertEqual(summary["status_counts"], {"predicted": 3, "rejected": 1})

    def test_confusion_matrix_places_predictions(self) -> None:
        results = [
            make_result(label="hello", predicted="hello"),
            make_result(label="hi", predicted="bye"),
            make_result(status="rejected", label="hi", predicted="", correct=False),
        ]
        confusion = ec.build_confusion_matrix(results, WORDS)
        hello, hi, bye = WORDS.index("hello"), WORDS.index("hi"), WORDS.index("bye")
        self.assertEqual(confusion[hello, hello], 1)
        self.assertEqual(confusion[hi, bye], 1)
        self.assertEqual(confusion.sum(), 2)

    def test_print_report_runs_without_raising(self) -> None:
        results = [
            make_result(speaker_id="spk_a", label="hello", predicted="hello"),
            make_result(speaker_id="spk_a", label="hi", predicted="bye", correct=False),
        ]
        ec.print_report(
            results, WORDS, source="npz", checkpoint=Path("fake_checkpoint.pt")
        )


class CsvOutputTests(unittest.TestCase):
    def test_csv_contains_one_row_per_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "predictions.csv"
            results = [
                make_result(sample_id="a", speaker_id="spk_a", label="hello", predicted="hello"),
                make_result(
                    sample_id="b",
                    speaker_id="spk_b",
                    label="hi",
                    repetition=2,
                    predicted="bye",
                    correct=False,
                    status="rejected",
                    note="too short",
                ),
            ]
            ec.write_results_csv(results, csv_path, source="mp4")

            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2)
        self.assertEqual(list(rows[0].keys()), ec.CSV_COLUMNS)
        self.assertEqual(rows[0]["sample_id"], "a")
        self.assertEqual(rows[0]["correct"], "True")
        self.assertEqual(rows[0]["source"], "mp4")
        self.assertEqual(rows[1]["status"], "rejected")
        self.assertEqual(rows[1]["note"], "too short")


class SingleVideoTests(unittest.TestCase):
    """CLI-level tests for the --video single-file prediction mode."""

    def _write_video(self, directory: Path, frame_count: int = 20) -> Path:
        video_path = directory / "clip.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48)
        )
        self.assertTrue(writer.isOpened())
        for _ in range(frame_count):
            writer.write(np.full((48, 64, 3), 100, dtype=np.uint8))
        writer.release()
        return video_path

    def _run_main(self, argv: list[str]) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.object(ec, "LipPredictor", FakePredictor),
            patch.object(ec, "MouthCropper", FakeCropper),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = ec.main(argv)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_video_mode_prints_prediction_and_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = self._write_video(Path(tmp))
            exit_code, stdout, stderr = self._run_main(
                ["--video", str(video_path), "--min-frames", "5"]
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Prediction: HELLO", stdout)
        self.assertIn("Word scores:", stdout)
        self.assertIn("hello:", stdout)
        self.assertIn("Mouth frames: 20", stdout)
        self.assertIn("fresh-detection rate: 100%", stdout)
        self.assertIn("mode: ctc", stdout)

    def test_video_mode_rejects_clips_failing_the_quality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = self._write_video(Path(tmp), frame_count=20)
            exit_code, stdout, stderr = self._run_main(
                ["--video", str(video_path), "--min-frames", "100"]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("Rejected:", stderr)
        self.assertIn("at least 100", stderr)

    def test_video_mode_reports_missing_file(self) -> None:
        exit_code, stdout, stderr = self._run_main(
            ["--video", "/nonexistent/clip.mp4"]
        )
        self.assertEqual(exit_code, 1)
        self.assertIn("Video file not found", stderr)

    def test_video_mode_rejects_dataset_only_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = self._write_video(Path(tmp))
            exit_code, stdout, stderr = self._run_main(
                ["--video", str(video_path), "--speakers", "spk_a"]
            )
        self.assertEqual(exit_code, 1)
        self.assertIn("cannot be combined with --speakers", stderr)


class MainTests(unittest.TestCase):
    def test_missing_checkpoint_returns_error_exit_code(self) -> None:
        exit_code = ec.main(["--checkpoint", "/nonexistent/checkpoint.pt"])
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
