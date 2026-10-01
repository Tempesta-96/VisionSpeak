"""Tests for the transformer temporal cores on the lip-reading models.

These lock in the invariants that make the swap safe:

* the CTC head still receives truthful, *emission-level* input lengths;
* every word in the custom dataset fits inside the committed CTC length budget;
* checkpoints trained before the transformer experiment still load strictly;
* zero-padded timesteps cannot corrupt valid ones through self-attention.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.LBY.model import (
    AttentionPooling,
    LipReadingClassifier,
    build_lip_reading_classifier,
)
from models.transformer_encoder import (
    TemporalTransformerEncoder,
    masked_mean_pool,
    propagate_lengths,
    strided_length,
)
from models.valente.model import (
    LipNet,
    architecture_from_checkpoint,
    build_lipnet,
    ctc_greedy_decode,
    tokens_to_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_WORDS_MANIFEST = PROJECT_ROOT / "data" / "custom_words" / "manifest.csv"
LEGACY_CTC_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "valente_custom_words_lipnet_ctc_0.852_acc.pt"
)
LEGACY_WORDS_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "custom_words.pt"

FRAME_SHAPE = (1, 50, 100)


def custom_word_labels() -> list[str]:
    """Read the word labels straight from the manifest (no pandas dependency)."""
    with CUSTOM_WORDS_MANIFEST.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
        label_column = header.index("label")
        return [line.rstrip("\n").split(",")[label_column] for line in handle if line.strip()]


def custom_word_frame_counts() -> list[int]:
    with CUSTOM_WORDS_MANIFEST.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
        column = header.index("frame_count")
        return [int(line.rstrip("\n").split(",")[column]) for line in handle if line.strip()]


class LengthPropagationTests(unittest.TestCase):
    def test_strided_length_matches_conv1d_floor_semantics(self) -> None:
        lengths = torch.tensor([45, 90, 92])
        self.assertEqual(strided_length(lengths, 1).tolist(), [45, 90, 92])
        self.assertEqual(strided_length(lengths, 2).tolist(), [23, 45, 46])

    def test_propagate_lengths_is_sequential_and_never_zero(self) -> None:
        lengths = torch.tensor([45, 90])
        self.assertEqual(propagate_lengths(lengths, (2, 1, 1)).tolist(), [23, 45])
        self.assertEqual(propagate_lengths(lengths, ()).tolist(), [45, 90])
        # A 2-frame clip still has to yield at least one emission.
        self.assertTrue(bool((propagate_lengths(torch.tensor([2]), (2,)) > 0).all()))

    def test_invalid_stride_rejected(self) -> None:
        with self.assertRaises(ValueError):
            strided_length(torch.tensor([10]), 0)


class TemporalEncoderTests(unittest.TestCase):
    def test_forward_shape_and_rejects_wrong_width(self) -> None:
        encoder = TemporalTransformerEncoder(model_dim=64, num_layers=2, num_heads=4)
        with torch.inference_mode():
            output = encoder(torch.randn(2, 20, 64))
        self.assertEqual(output.shape, (2, 20, 64))
        with self.assertRaises(ValueError):
            encoder(torch.randn(2, 20, 32))
        with self.assertRaises(ValueError):
            encoder(torch.randn(2, 20))

    def test_max_tokens_ceiling_is_enforced(self) -> None:
        encoder = TemporalTransformerEncoder(model_dim=32, num_heads=4, max_tokens=16)
        with torch.inference_mode():
            encoder(torch.randn(1, 16, 32))
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            encoder(torch.randn(1, 17, 32))

    def test_masked_positions_do_not_influence_valid_ones(self) -> None:
        """Self-attention must not leak zero-padded timesteps into valid ones."""
        torch.manual_seed(0)
        encoder = TemporalTransformerEncoder(model_dim=64, num_layers=2, num_heads=4)
        encoder.eval()
        valid = torch.randn(1, 30, 64)
        padded = torch.zeros(1, 60, 64)
        padded[:, :30] = valid

        with torch.inference_mode():
            short = encoder(valid, torch.tensor([30]))
            long = encoder(padded, torch.tensor([30]))
            unmasked = encoder(padded)

        masked_gap = (long[:, :30] - short).abs().max().item()
        unmasked_gap = (unmasked[:, :30] - short).abs().max().item()
        self.assertLess(masked_gap, unmasked_gap)
        self.assertLess(masked_gap, 1e-4)

    def test_lengths_longer_than_sequence_are_rejected(self) -> None:
        encoder = TemporalTransformerEncoder(model_dim=32, num_heads=4)
        with self.assertRaises(ValueError):
            encoder(torch.randn(1, 10, 32), torch.tensor([11]))

    def test_configuration_validation(self) -> None:
        with self.assertRaises(ValueError):
            TemporalTransformerEncoder(model_dim=30, num_heads=4)  # not divisible
        with self.assertRaises(ValueError):
            TemporalTransformerEncoder(num_layers=0)
        with self.assertRaises(ValueError):
            TemporalTransformerEncoder(dropout=1.0)


class MaskedPoolingTests(unittest.TestCase):
    def test_masked_mean_ignores_padding(self) -> None:
        features = torch.zeros(1, 10, 4)
        features[:, :3] = 2.0
        features[:, 3:] = 100.0  # would dominate an unmasked mean
        pooled = masked_mean_pool(features, torch.tensor([3]))
        torch.testing.assert_close(pooled, torch.full((1, 4), 2.0))

    def test_unmasked_mean_matches_torch_mean(self) -> None:
        features = torch.randn(2, 7, 5)
        torch.testing.assert_close(masked_mean_pool(features), features.mean(dim=1))

    def test_attention_pooling_ignores_padding(self) -> None:
        torch.manual_seed(0)
        pool = AttentionPooling(model_dim=8).eval()
        features = torch.randn(1, 20, 8)
        padded = torch.zeros(1, 40, 8)
        padded[:, :20] = features
        with torch.inference_mode():
            short = pool(features, torch.tensor([20]))
            long = pool(padded, torch.tensor([20]))
        torch.testing.assert_close(short, long, atol=1e-5, rtol=1e-5)

    def test_attention_pooling_never_returns_nan(self) -> None:
        pool = AttentionPooling(model_dim=4).eval()
        with torch.inference_mode():
            output = pool(torch.randn(1, 5, 4), torch.tensor([1]))
        self.assertFalse(bool(torch.isnan(output).any()))


class LipNetCTCTests(unittest.TestCase):
    def _model(self, architecture: str, num_classes: int = 15) -> LipNet:
        model = build_lipnet({"architecture": architecture}, num_classes=num_classes)
        # feature_projection is lazy; materialise it with one forward pass.
        with torch.inference_mode():
            model(torch.zeros(1, 10, *FRAME_SHAPE), torch.tensor([10]))
        return model

    def test_ctc_log_probs_layout_and_emission_lengths(self) -> None:
        model = self._model("transformer").eval()
        frames = torch.randn(2, 90, *FRAME_SHAPE)
        lengths = torch.tensor([90, 45])
        with torch.inference_mode():
            log_probs, input_lengths = model.ctc_log_probs(frames, lengths)

        self.assertEqual(log_probs.dim(), 3)
        self.assertEqual(log_probs.shape[1], 2)  # [time, batch, classes]
        self.assertEqual(log_probs.shape[0], 45)  # time halved by the frontend
        self.assertEqual(input_lengths.tolist(), [45, 23])
        # Log-probabilities must normalise over the class axis.
        torch.testing.assert_close(
            log_probs.exp().sum(dim=-1),
            torch.ones_like(log_probs[..., 0]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_bigru_path_keeps_full_temporal_resolution(self) -> None:
        model = self._model("bigru").eval()
        frames = torch.randn(1, 90, *FRAME_SHAPE)
        with torch.inference_mode():
            logits = model(frames, torch.tensor([90]))
        self.assertEqual(logits.shape[1], 90)
        self.assertEqual(model.emission_lengths(torch.tensor([90])).tolist(), [90])

    def test_ctc_length_budget_holds_for_every_recorded_clip(self) -> None:
        """CTC needs input_length >= 2 * target_length + 1.

        The longest custom word is 'thanks' (6 characters) and the shortest
        recorded clip is 45 frames, so the transformer frontend must leave at
        least 13 emissions for that clip.
        """
        labels = custom_word_labels()
        frame_counts = custom_word_frame_counts()
        self.assertTrue(labels and frame_counts, "manifest should not be empty")

        longest_word = max(len(word) for word in labels)
        shortest_clip = min(frame_counts)
        required = 2 * longest_word + 1

        model = self._model("transformer").eval()
        lengths = torch.tensor([shortest_clip])
        with torch.inference_mode():
            log_probs, input_lengths = model.ctc_log_probs(
                torch.randn(1, shortest_clip, *FRAME_SHAPE), lengths
            )

        self.assertEqual(longest_word, 6)
        self.assertGreaterEqual(
            input_lengths.item(),
            required,
            f"only {input_lengths.item()} emissions for a {longest_word}-character "
            f"word; CTC needs at least {required}",
        )
        self.assertGreaterEqual(log_probs.shape[0], required)

    def test_ctc_loss_is_finite_on_a_variable_length_batch(self) -> None:
        model = self._model("transformer").eval()
        batch = torch.zeros(2, 90, *FRAME_SHAPE)
        batch[0, :90] = torch.randn(90, *FRAME_SHAPE)
        batch[1, :50] = torch.randn(50, *FRAME_SHAPE)
        targets = torch.tensor([1, 2, 3, 1, 2, 3, 4])
        target_lengths = torch.tensor([3, 4])

        with torch.inference_mode():
            log_probs, input_lengths = model.ctc_log_probs(
                batch, torch.tensor([90, 50])
            )
        loss = nn.CTCLoss(blank=0, zero_infinity=True)(
            log_probs, targets, input_lengths, target_lengths
        )
        self.assertTrue(torch.isfinite(loss))

    def test_padded_batch_does_not_corrupt_the_solo_clip(self) -> None:
        """A zero-padded batch member must match its standalone forward pass."""
        torch.manual_seed(1)
        model = self._model("transformer").eval()
        clip = torch.randn(1, 45, *FRAME_SHAPE)

        with torch.inference_mode():
            solo, _ = model.ctc_log_probs(clip, torch.tensor([45]))

        batch = torch.zeros(2, 90, *FRAME_SHAPE)
        batch[0, :45] = clip[0]
        batch[1] = torch.randn(90, *FRAME_SHAPE)
        with torch.inference_mode():
            padded, _ = model.ctc_log_probs(batch, torch.tensor([45, 90]))

        emissions = solo.size(0)
        gap = (padded[:emissions, 0] - solo[:, 0]).abs().max().item()
        scale = solo[:, 0].std().item()
        # The residual is the STCNN's own temporal convolution, which the
        # BiGRU model shows too; the transformer must not be materially worse.
        self.assertLess(gap / scale, 0.5)

    def test_logits_are_class_width(self) -> None:
        model = self._model("transformer", num_classes=15).eval()
        with torch.inference_mode():
            logits = model(torch.randn(1, 90, *FRAME_SHAPE), torch.tensor([90]))
        self.assertEqual(logits.shape, (1, 45, 15))

    def test_strided_length_formula_matches_real_conv_output(self) -> None:
        """The predicted emission count must equal what the frontend produces.

        If this drifts, CTC silently receives lengths that do not describe the
        logits it is scoring.
        """
        conv = nn.Conv3d(
            1, 2, kernel_size=(3, 5, 5), stride=(2, 2, 2), padding=(1, 2, 2)
        )
        with torch.inference_mode():
            for frames in range(1, 40):
                real = conv(torch.zeros(1, 1, frames, 32, 32)).shape[2]
                predicted = strided_length(torch.tensor([frames]), 2).item()
                self.assertEqual(
                    predicted,
                    real,
                    f"frames={frames}: formula predicted {predicted}, conv gave {real}",
                )

    def test_emission_lengths_match_the_frontend_for_recorded_lengths(self) -> None:
        """Check the whole stacked frontend, not just one conv layer."""
        model = self._model("transformer").eval()
        observed = sorted(set(custom_word_frame_counts()))
        self.assertTrue(observed)
        for frames in observed:
            with torch.inference_mode():
                logits = model(
                    torch.randn(1, frames, *FRAME_SHAPE), torch.tensor([frames])
                )
            predicted = model.emission_lengths(torch.tensor([frames])).item()
            self.assertEqual(
                predicted,
                logits.shape[1],
                f"frames={frames}: predicted {predicted} emissions, "
                f"frontend produced {logits.shape[1]}",
            )

    def test_infer_input_lengths_reads_zero_padding(self) -> None:
        batch = torch.zeros(2, 60, *FRAME_SHAPE)
        batch[0] = torch.randn(60, *FRAME_SHAPE)
        batch[1, :30] = torch.randn(30, *FRAME_SHAPE)
        self.assertEqual(LipNet.infer_input_lengths(batch).tolist(), [60, 30])

    def test_unknown_architecture_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LipNet(num_classes=10, architecture="conformer")

    def test_greedy_decode_respects_emission_lengths(self) -> None:
        log_probs = torch.zeros(10, 1, 3)
        log_probs[0, 0, 1] = 5.0
        log_probs[1, 0, 1] = 5.0
        log_probs[2, 0, 0] = 5.0
        tokens = ctc_greedy_decode(log_probs, torch.tensor([5]), blank_index=0)
        self.assertEqual(tokens[0][:1], [1])
        self.assertEqual(tokens_to_text([1], {1: "h"}), "h")


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_legacy_ctc_checkpoint_loads_strictly_as_bigru(self) -> None:
        checkpoint = torch.load(
            LEGACY_CTC_CHECKPOINT, map_location="cpu", weights_only=False
        )
        self.assertEqual(architecture_from_checkpoint(checkpoint), "bigru")

        num_classes = len(checkpoint["char_to_index"]) + 1
        model = build_lipnet(checkpoint, num_classes=num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])  # strict by design
        model.eval()

        with torch.inference_mode():
            log_probs, input_lengths = model.ctc_log_probs(
                torch.randn(1, 90, *FRAME_SHAPE), torch.tensor([90])
            )
        self.assertEqual(log_probs.shape[0], 90)
        self.assertEqual(input_lengths.tolist(), [90])

    def test_legacy_words_checkpoint_loads_strictly_as_bigru(self) -> None:
        checkpoint = torch.load(
            LEGACY_WORDS_CHECKPOINT, map_location="cpu", weights_only=False
        )
        model = build_lip_reading_classifier(
            checkpoint, num_classes=len(checkpoint["label_to_index"])
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        with torch.inference_mode():
            logits = model(torch.randn(2, 60, *FRAME_SHAPE))
        self.assertEqual(logits.shape, (2, len(checkpoint["label_to_index"])))

    def test_missing_architecture_key_defaults_to_bigru(self) -> None:
        self.assertEqual(architecture_from_checkpoint({}), "bigru")
        self.assertEqual(
            architecture_from_checkpoint({"architecture": "TRANSFORMER"}),
            "transformer",
        )

    def test_extra_checkpoint_metadata_is_tolerated(self) -> None:
        model = build_lipnet(
            {"architecture": "transformer", "epoch": 135, "some_future_key": [1, 2]},
            num_classes=12,
        )
        self.assertEqual(model.architecture, "transformer")

    def test_non_mapping_settings_rejected(self) -> None:
        with self.assertRaises(TypeError):
            build_lipnet(["not", "a", "dict"], num_classes=10)
        with self.assertRaises(TypeError):
            build_lip_reading_classifier("nope", num_classes=10)


class TransformerConfigurationTests(unittest.TestCase):
    @staticmethod
    def _parameter_count(module: nn.Module) -> int:
        with torch.inference_mode():
            module(torch.zeros(1, 10, *FRAME_SHAPE), torch.tensor([10]))
        return sum(p.numel() for p in module.parameters())

    def test_default_parameter_budget_stays_near_the_bigru_baseline(self) -> None:
        """The default transformer must not silently explode or collapse capacity.

        The shipped default is d=256 with 3 encoder layers, which keeps the
        width identical to the BiGRU baseline (2 x 256 bidirectional = 512) and
        lands within ~15% of its parameter count. That makes the comparison a
        fair test of attention versus recurrence rather than of raw capacity.
        """
        transformer = LipNet(num_classes=15, architecture="transformer")
        baseline = LipNet(num_classes=15, architecture="bigru")

        transformer_params = self._parameter_count(transformer)
        baseline_params = self._parameter_count(baseline)

        self.assertEqual(baseline_params, 2_621_743)
        self.assertEqual(transformer.hidden_size, baseline.hidden_size)
        self.assertLess(abs(transformer_params - baseline_params) / baseline_params, 0.30)

    def test_gru_core_worth_of_capacity_is_replaced_by_attention(self) -> None:
        """The transformer must carry GRU parameters only on the bigru path."""
        transformer = LipNet(num_classes=15, architecture="transformer")
        self.assertFalse([name for name in transformer.state_dict() if "gru" in name])
        self.assertTrue(
            [name for name in transformer.state_dict() if name.startswith("encoder.")]
        )

        baseline = LipNet(num_classes=15, architecture="bigru")
        self.assertTrue([name for name in baseline.state_dict() if "gru" in name])
        self.assertFalse(
            [name for name in baseline.state_dict() if name.startswith("encoder.")]
        )

    def test_transformer_only_strides_time_in_the_first_block(self) -> None:
        transformer = LipNet(num_classes=15, architecture="transformer")
        baseline = LipNet(num_classes=15, architecture="bigru")
        self.assertEqual(transformer.temporal_strides, (2, 1, 1))
        self.assertEqual(baseline.temporal_strides, (1, 1, 1))
        self.assertEqual(transformer.stcnn[0][0].stride, (2, 2, 2))
        self.assertEqual(baseline.stcnn[0][0].stride, (1, 2, 2))

    def test_classifier_transformer_still_returns_one_logit_vector(self) -> None:
        model = LipReadingClassifier(
            num_classes=8, architecture="transformer", pooling="attention"
        ).eval()
        with torch.inference_mode():
            logits = model(torch.randn(3, 70, *FRAME_SHAPE))
        self.assertEqual(logits.shape, (3, 8))

    def test_lengths_are_optional_and_backward_compatible(self) -> None:
        model = LipReadingClassifier(num_classes=8, architecture="transformer").eval()
        with torch.inference_mode():
            logits = model(torch.randn(2, 40, *FRAME_SHAPE))
        self.assertEqual(logits.shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
