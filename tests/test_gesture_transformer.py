"""Tests for the transformer temporal core on the hand-gesture model.

The point of this branch is that the gesture model finally sees motion, so the
tests concentrate on the things that make that true or false:

* a real window produces a different result from a single repeated frame;
* padding inside a window cannot change the prediction;
* the live frame buffer provides context, but never stale context;
* legacy single-frame checkpoints keep working unchanged.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import torch

from inference.sx_worker import FrameWindowBuffer, SXDetector
from models.SX.inference import load_sx_checkpoint
from models.SX.model import HandSignLSTM
from models.SX.transformer import GestureTransformer, build_gesture_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_HANDS_ONLY = PROJECT_ROOT / "checkpoints" / "sx_hand_sign_cached_both_hands.pt"
LEGACY_HANDS_BODY = PROJECT_ROOT / "checkpoints" / "sx_hand_sign_cached_both_hands_body.pt"


def write_transformer_checkpoint(
    path: Path,
    *,
    input_size: int = 225,
    window_size: int = 8,
    num_classes: int = 4,
    architecture: str = "transformer",
    **overrides,
) -> None:
    """Persist a checkpoint shaped like the ones models/SX/train.py writes."""
    settings = {
        "input_size": input_size,
        "architecture": architecture,
        "num_layers": 2,
        "dropout": 0.2,
        "d_model": 64,
        "num_heads": 4,
        "feedforward_dim": 128,
        "pooling": "cls",
        "max_tokens": window_size + 1,
        "window_size": window_size,
        "sequence_length": window_size,
        "two_hands": True,
        "use_body_pose": input_size == 225,
        "num_classes": num_classes,
        "class_to_idx": {f"c{i}": i for i in range(num_classes)},
        "idx_to_class": {i: f"c{i}" for i in range(num_classes)},
    }
    settings.update(overrides)
    model = build_gesture_model(settings, num_classes=num_classes)
    torch.save({**settings, "model_state_dict": model.state_dict()}, path)


class GestureTransformerShapeTests(unittest.TestCase):
    def test_window_dimensions(self) -> None:
        model = GestureTransformer(input_size=225, num_classes=18, max_tokens=17).eval()
        for window in (1, 2, 16):
            with torch.inference_mode():
                logits = model(torch.randn(3, window, 225))
            self.assertEqual(logits.shape, (3, 18))

    def test_all_supported_landmark_layouts(self) -> None:
        for input_size in (63, 126, 225):
            model = GestureTransformer(
                input_size=input_size, num_classes=5, max_tokens=17
            ).eval()
            with torch.inference_mode():
                logits = model(torch.randn(2, 16, input_size))
            self.assertEqual(logits.shape, (2, 5))

    def test_single_frame_is_still_accepted(self) -> None:
        """Backward compatibility: window_size=1 must not crash."""
        model = GestureTransformer(input_size=126, num_classes=18, max_tokens=2).eval()
        with torch.inference_mode():
            logits = model(torch.randn(1, 1, 126))
        self.assertEqual(logits.shape, (1, 18))
        self.assertAlmostEqual(logits.softmax(dim=1).sum().item(), 1.0, places=5)

    def test_window_larger_than_max_tokens_is_rejected(self) -> None:
        model = GestureTransformer(input_size=126, num_classes=4, max_tokens=5)
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            model(torch.randn(1, 5, 126))

    def test_invalid_shapes_and_configuration_rejected(self) -> None:
        model = GestureTransformer(input_size=126, num_classes=4)
        with self.assertRaises(ValueError):
            model(torch.randn(16, 126))  # missing batch dim
        with self.assertRaises(ValueError):
            model(torch.randn(1, 16, 99))  # wrong feature width
        with self.assertRaises(ValueError):
            GestureTransformer(input_size=126, d_model=64, num_heads=5)
        with self.assertRaises(ValueError):
            GestureTransformer(input_size=126, pooling="max")
        with self.assertRaises(ValueError):
            GestureTransformer(input_size=126, max_tokens=1)

    def test_mean_pooling_also_works(self) -> None:
        model = GestureTransformer(
            input_size=126, num_classes=4, pooling="mean", max_tokens=17
        ).eval()
        with torch.inference_mode():
            logits = model(torch.randn(2, 16, 126))
        self.assertEqual(logits.shape, (2, 4))


class WindowMotionTests(unittest.TestCase):
    """A window is only worth it if the model can use the time axis."""

    def test_a_moving_window_differs_from_a_repeated_frame(self) -> None:
        torch.manual_seed(0)
        model = GestureTransformer(input_size=126, num_classes=4, max_tokens=17).eval()
        still = torch.randn(1, 1, 126).repeat(1, 16, 1)
        moving = still.clone()
        moving[:, 8:] += 2.5  # a real change part-way through the window

        with torch.inference_mode():
            still_logits = model(still)
            moving_logits = model(moving)

        self.assertGreater(
            (still_logits - moving_logits).abs().max().item(),
            1e-3,
            "the model ignored the time axis: a moving window scored like a still one",
        )

    def test_padded_positions_do_not_change_the_prediction(self) -> None:
        torch.manual_seed(0)
        model = GestureTransformer(input_size=126, num_classes=4, max_tokens=17).eval()
        real = torch.randn(1, 6, 126)
        padded = torch.zeros(1, 16, 126)
        padded[:, :6] = real

        with torch.inference_mode():
            short = model(real, torch.tensor([6]))
            masked = model(padded, torch.tensor([6]))
            unmasked = model(padded)

        torch.testing.assert_close(short, masked, atol=1e-5, rtol=1e-5)
        self.assertGreater((short - unmasked).abs().max().item(), 1e-4)

    def test_lengths_longer_than_window_rejected(self) -> None:
        model = GestureTransformer(input_size=126, num_classes=4, max_tokens=17)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 8, 126), torch.tensor([9]))


class FrameWindowBufferTests(unittest.TestCase):
    def test_empty_buffer_has_no_tensor(self) -> None:
        buffer = FrameWindowBuffer(4, 3)
        self.assertIsNone(buffer.as_tensor())
        self.assertFalse(buffer.is_full())

    def test_partial_window_repeats_the_newest_frame(self) -> None:
        buffer = FrameWindowBuffer(4, 3)
        buffer.append([1.0, 2.0, 3.0])
        window = buffer.as_tensor()
        self.assertEqual(tuple(window.shape), (1, 4, 3))
        self.assertEqual(window[0, :, 0].tolist(), [1.0, 1.0, 1.0, 1.0])

    def test_full_window_keeps_chronological_order(self) -> None:
        buffer = FrameWindowBuffer(4, 1)
        for value in (1.0, 2.0, 3.0, 4.0):
            buffer.append([value])
        self.assertTrue(buffer.is_full())
        self.assertEqual(buffer.as_tensor()[0, :, 0].tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_oldest_frame_is_evicted(self) -> None:
        buffer = FrameWindowBuffer(3, 1)
        for value in (1.0, 2.0, 3.0, 4.0):
            buffer.append([value])
        self.assertEqual(buffer.as_tensor()[0, :, 0].tolist(), [2.0, 3.0, 4.0])

    def test_wrong_feature_width_is_rejected(self) -> None:
        buffer = FrameWindowBuffer(4, 3)
        with self.assertRaises(ValueError):
            buffer.append([1.0, 2.0])

    def test_clear_resets_the_window(self) -> None:
        buffer = FrameWindowBuffer(3, 1)
        buffer.append([1.0])
        buffer.clear()
        self.assertIsNone(buffer.as_tensor())
        self.assertEqual(len(buffer), 0)

    def test_invalid_construction_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FrameWindowBuffer(0, 3)
        with self.assertRaises(ValueError):
            FrameWindowBuffer(3, 0)


class SXDetectorBufferTests(unittest.TestCase):
    """The live loop must feed windows without ever using stale context."""

    def _detector(self, path: Path) -> SXDetector:
        detector = SXDetector(path, device="cpu")
        detector.hands, detector.pose = Mock(), Mock()
        detector.model = Mock(return_value=torch.zeros(1, 4))
        return detector

    def test_windowed_checkpoint_feeds_a_full_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windowed.pt"
            write_transformer_checkpoint(path, input_size=225, window_size=16, num_classes=4)
            detector = self._detector(path)
            self.assertEqual(detector.config.window_size, 16)
            self.assertEqual(detector.config.architecture, "transformer")

            vector = np.zeros(225, dtype=np.float32)
            hand = [NS(landmark=[NS(x=0.5, y=0.5) for _ in range(21)])]
            with patch("inference.sx_worker.extract_rgb", return_value=(vector, hand)):
                for _ in range(3):
                    detector.predict_frame(np.zeros((240, 320, 3), dtype=np.uint8))

            sent = detector.model.call_args.args[0]
            self.assertEqual(tuple(sent.shape), (1, 16, 225))

    def test_legacy_checkpoint_still_sends_one_frame(self) -> None:
        detector = self._detector(LEGACY_HANDS_BODY)
        self.assertEqual(detector.config.window_size, 1)

        vector = np.zeros(225, dtype=np.float32)
        hand = [NS(landmark=[NS(x=0.5, y=0.5) for _ in range(21)])]
        with patch("inference.sx_worker.extract_rgb", return_value=(vector, hand)):
            detector.predict_frame(np.zeros((240, 320, 3), dtype=np.uint8))

        self.assertEqual(tuple(detector.model.call_args.args[0].shape), (1, 1, 225))

    def test_no_hand_clears_stale_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windowed.pt"
            write_transformer_checkpoint(path, input_size=225, window_size=8, num_classes=4)
            detector = self._detector(path)

            vector = np.zeros(225, dtype=np.float32)
            hand = [NS(landmark=[NS(x=0.5, y=0.5) for _ in range(21)])]
            with patch("inference.sx_worker.extract_rgb", return_value=(vector, hand)):
                for _ in range(4):
                    detector.predict_frame(np.zeros((240, 320, 3), dtype=np.uint8))
            self.assertEqual(len(detector.buffer), 4)

            with patch("inference.sx_worker.extract_rgb", return_value=(vector, None)):
                label, confidence, _ = detector.predict_frame(
                    np.zeros((240, 320, 3), dtype=np.uint8)
                )

            self.assertEqual(label, "No hand detected")
            self.assertEqual(confidence, 0.0)
            self.assertEqual(len(detector.buffer), 0)

    def test_render_returns_the_frame_shape(self) -> None:
        detector = self._detector(LEGACY_HANDS_BODY)
        vector = np.zeros(225, dtype=np.float32)
        hand = [NS(landmark=[NS(x=0.5, y=0.5) for _ in range(21)])]
        with patch("inference.sx_worker.extract_rgb", return_value=(vector, hand)):
            _, _, render = detector.predict_frame(np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertEqual(render().shape, (240, 320, 3))


class GestureCheckpointTests(unittest.TestCase):
    def test_legacy_checkpoints_load_as_single_frame_lstm(self) -> None:
        for path, size in ((LEGACY_HANDS_ONLY, 126), (LEGACY_HANDS_BODY, 225)):
            model, labels, config = load_sx_checkpoint(path, torch.device("cpu"))
            self.assertIsInstance(model, HandSignLSTM)
            self.assertEqual(config.architecture, "lstm")
            self.assertEqual(config.window_size, 1)
            self.assertEqual(config.sequence_length, 1)  # deprecated alias
            self.assertEqual(config.input_size, size)
            self.assertEqual(len(labels), 18)

    def test_transformer_checkpoint_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.pt"
            write_transformer_checkpoint(path, input_size=225, window_size=12, num_classes=6)
            model, labels, config = load_sx_checkpoint(path, torch.device("cpu"))

            self.assertIsInstance(model, GestureTransformer)
            self.assertEqual(config.architecture, "transformer")
            self.assertEqual(config.window_size, 12)
            self.assertEqual(len(labels), 6)
            with torch.inference_mode():
                logits = model(torch.randn(1, 12, 225))
            self.assertEqual(logits.shape, (1, 6))

    def test_lstm_checkpoint_with_a_window_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            write_transformer_checkpoint(
                path, input_size=126, window_size=4, num_classes=4, architecture="lstm"
            )
            with self.assertRaisesRegex(ValueError, "window_size=1"):
                load_sx_checkpoint(path, torch.device("cpu"))

    def test_window_size_and_sequence_length_must_agree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clash.pt"
            write_transformer_checkpoint(
                path, input_size=126, window_size=6, num_classes=4, sequence_length=3
            )
            with self.assertRaisesRegex(ValueError, "disagree"):
                load_sx_checkpoint(path, torch.device("cpu"))

    def test_max_tokens_too_small_for_the_window_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.pt"
            write_transformer_checkpoint(
                path, input_size=126, window_size=8, num_classes=4, max_tokens=8
            )
            with self.assertRaisesRegex(ValueError, "max_tokens"):
                load_sx_checkpoint(path, torch.device("cpu"))

    def test_architecture_mismatch_reports_the_recorded_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mismatch.pt"
            write_transformer_checkpoint(path, input_size=126, window_size=4, num_classes=4)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            # Transformer weights declared as an LSTM at window_size=1.
            checkpoint["architecture"] = "lstm"
            checkpoint["window_size"] = 1
            checkpoint["sequence_length"] = 1
            torch.save(checkpoint, path)

            with self.assertRaisesRegex(ValueError, "lstm"):
                load_sx_checkpoint(path, torch.device("cpu"))

    def test_unknown_architecture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future.pt"
            # Hand-written so the loader, not the factory, is under test here.
            torch.save(
                {
                    "model_state_dict": {},
                    "class_to_idx": {f"c{i}": i for i in range(4)},
                    "idx_to_class": {i: f"c{i}" for i in range(4)},
                    "input_size": 126,
                    "num_classes": 4,
                    "architecture": "conformer",
                    "window_size": 4,
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_sx_checkpoint(path, torch.device("cpu"))


class GestureFactoryTests(unittest.TestCase):
    def test_factory_defaults_to_the_legacy_lstm(self) -> None:
        model = build_gesture_model({"input_size": 126}, num_classes=18)
        self.assertIsInstance(model, HandSignLSTM)

    def test_factory_builds_the_transformer(self) -> None:
        model = build_gesture_model(
            {"input_size": 225, "architecture": "transformer", "max_tokens": 9},
            num_classes=6,
        )
        self.assertIsInstance(model, GestureTransformer)
        self.assertEqual(model.input_size, 225)

    def test_factory_rejects_unknown_architecture_and_bad_settings(self) -> None:
        with self.assertRaises(ValueError):
            build_gesture_model({"input_size": 126, "architecture": "conformer"}, 4)
        with self.assertRaises(TypeError):
            build_gesture_model("not-a-mapping", 4)
        with self.assertRaises(KeyError):
            build_gesture_model({"architecture": "transformer"}, 4)  # no input_size


if __name__ == "__main__":
    unittest.main()
