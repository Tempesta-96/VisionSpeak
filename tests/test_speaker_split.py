"""Tests for the speaker-disjoint evaluation split.

The default per-clip split puts every speaker in both train and test, so its
accuracy measures same-speaker recall rather than generalisation. These tests
pin down the property that makes the speaker-disjoint split trustworthy: no
speaker may appear on both sides.
"""

from __future__ import annotations

import unittest

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.prepare_custom_word_dataset import (
    DEFAULT_DATA_DIR,
    available_speakers,
    choose_holdout_speaker,
    collate_fn,
    get_custom_words_dataloaders,
    load_manifest,
    speaker_disjoint_split,
    stratified_split,
)

WORDS = ("bad", "bye", "good", "hello", "hi", "no", "thanks", "yes")


def synthetic_manifest(
    speakers: tuple[str, ...] = ("s1", "s2", "s3"),
    words: tuple[str, ...] = WORDS,
    per_word: int = 3,
) -> pd.DataFrame:
    rows = []
    for speaker in speakers:
        for word in words:
            for repetition in range(per_word):
                rows.append(
                    {
                        "sample_id": f"{speaker}_{word}_{repetition}",
                        "speaker_id": speaker,
                        "label": word,
                        "repetition": repetition,
                        "relative_path": f"clips/{speaker}/{word}/{repetition}.npz",
                    }
                )
    return pd.DataFrame(rows)


class RealManifestSplitTests(unittest.TestCase):
    """Properties that must hold on the actual recorded dataset."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest(DEFAULT_DATA_DIR)

    def test_default_split_leaks_speakers(self) -> None:
        """Document the weakness of the default split rather than assume it."""
        train_df, test_df = stratified_split(self.manifest, 0.2, 42)
        overlap = set(train_df["speaker_id"]) & set(test_df["speaker_id"])
        self.assertTrue(
            overlap,
            "the per-clip split is expected to leak speakers; if this fails the "
            "README warning is now wrong",
        )

    def test_speaker_disjoint_split_has_no_overlap_for_every_speaker(self) -> None:
        for speaker in available_speakers(self.manifest):
            with self.subTest(speaker=speaker):
                train_df, test_df, chosen = speaker_disjoint_split(
                    self.manifest, test_speaker=speaker
                )
                self.assertEqual(chosen, speaker)
                self.assertEqual(set(train_df["speaker_id"]), set(available_speakers(self.manifest)) - {speaker})
                self.assertEqual(set(test_df["speaker_id"]), {speaker})
                self.assertFalse(set(train_df["speaker_id"]) & set(test_df["speaker_id"]))

    def test_every_held_out_speaker_covers_all_words(self) -> None:
        for speaker in available_speakers(self.manifest):
            with self.subTest(speaker=speaker):
                _, test_df, _ = speaker_disjoint_split(self.manifest, test_speaker=speaker)
                self.assertEqual(set(test_df["label"]), set(WORDS))

    def test_held_out_speaker_is_absent_from_training(self) -> None:
        holdout = choose_holdout_speaker(self.manifest)
        self.assertIn(holdout, available_speakers(self.manifest))
        train_df, _, _ = speaker_disjoint_split(self.manifest, test_speaker=holdout)
        self.assertNotIn(holdout, set(train_df["speaker_id"]))

    def test_dataloaders_honour_speaker_holdout(self) -> None:
        train_loader, test_loader, label_to_index, index_to_label = (
            get_custom_words_dataloaders(batch_size=4, speaker_holdout="TestID")
        )
        self.assertEqual(set(train_loader.dataset.dataframe["speaker_id"]),
                         set(available_speakers(self.manifest)) - {"TestID"})
        self.assertNotIn("TestID", set(train_loader.dataset.dataframe["speaker_id"]))
        self.assertEqual(
            set(test_loader.dataset.dataframe["speaker_id"]), {"TestID"}
        )

        frames, labels = next(iter(test_loader))
        self.assertEqual(frames.dim(), 5)
        self.assertEqual(frames.shape[2], 1)  # grayscale channel
        self.assertEqual(frames.shape[3:], (50, 100))
        self.assertEqual(labels.dtype, torch.long)
        self.assertEqual(set(index_to_label), set(label_to_index.values()))

    def test_speaker_holdout_keeps_the_word_vocabulary_complete(self) -> None:
        """The vocab is derived from the whole manifest, so it must not shrink."""
        _, _, full_vocab, _ = get_custom_words_dataloaders(batch_size=4)
        _, _, holdout_vocab, _ = get_custom_words_dataloaders(
            batch_size=4, speaker_holdout="TestID"
        )
        self.assertEqual(full_vocab, holdout_vocab)
        self.assertEqual(set(full_vocab), set(WORDS))


class SplitValidationTests(unittest.TestCase):
    def test_unknown_speaker_is_rejected(self) -> None:
        manifest = synthetic_manifest()
        with self.assertRaisesRegex(ValueError, "Unknown speaker"):
            speaker_disjoint_split(manifest, test_speaker="nope")

    def test_single_speaker_manifest_is_rejected(self) -> None:
        manifest = synthetic_manifest(speakers=("only",))
        with self.assertRaisesRegex(ValueError, "at least two speakers"):
            speaker_disjoint_split(manifest)

    def test_pruned_manifest_holding_out_its_only_speaker_is_rejected(self) -> None:
        """A word with no training example must fail loudly, not silently."""
        manifest = pd.DataFrame(
            [
                # "bye" is spoken only by s1 and no other speaker.
                {"speaker_id": "s1", "label": "bye", "relative_path": "a.npz"},
                {"speaker_id": "s1", "label": "yes", "relative_path": "b.npz"},
                {"speaker_id": "s2", "label": "yes", "relative_path": "c.npz"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "no training examples"):
            speaker_disjoint_split(manifest, test_speaker="s1")

    def test_choose_holdout_prefers_full_coverage(self) -> None:
        manifest = pd.DataFrame(
            [
                {"speaker_id": "sparse", "label": "yes", "relative_path": "a.npz"},
                {"speaker_id": "rich", "label": "yes", "relative_path": "b.npz"},
                {"speaker_id": "rich", "label": "no", "relative_path": "c.npz"},
            ]
        )
        self.assertEqual(choose_holdout_speaker(manifest), "rich")

    def test_choose_holdout_is_deterministic(self) -> None:
        manifest = synthetic_manifest()
        self.assertEqual(
            choose_holdout_speaker(manifest), choose_holdout_speaker(manifest)
        )

    def test_split_result_is_index_reset_and_accounts_for_every_clip(self) -> None:
        manifest = synthetic_manifest()
        train_df, validation_df, test_df, _ = speaker_disjoint_split(
            manifest, return_validation=True
        )
        for frame in (train_df, validation_df, test_df):
            self.assertEqual(list(frame.index), list(range(len(frame))))
        # Validation is carved out of the non-held-out speakers, so the three
        # frames must partition the manifest exactly once.
        self.assertEqual(
            len(train_df) + len(validation_df) + len(test_df), len(manifest)
        )
        self.assertTrue(set(train_df["speaker_id"]).isdisjoint(set(test_df["speaker_id"])))
        self.assertTrue(
            set(validation_df["speaker_id"]).isdisjoint(set(test_df["speaker_id"]))
        )

    def test_return_validation_is_opt_in(self) -> None:
        manifest = synthetic_manifest()
        self.assertEqual(len(speaker_disjoint_split(manifest)), 3)
        self.assertEqual(
            len(speaker_disjoint_split(manifest, return_validation=True)), 4
        )


class CarveValidationTests(unittest.TestCase):
    """``carve_validation=False`` spends every non-test clip on training.

    The point of the mode is that nothing is held back for model selection, so
    the held-out speaker stays unread until the run is over. These tests pin the
    two properties that makes true: training gets strictly more clips, and the
    test speaker is still absent from it.
    """

    def test_no_validation_frame_is_carved_out(self) -> None:
        manifest = synthetic_manifest()
        train_df, validation_df, test_df, chosen = speaker_disjoint_split(
            manifest,
            test_speaker="s1",
            return_validation=True,
            carve_validation=False,
        )
        self.assertEqual(chosen, "s1")
        self.assertTrue(validation_df.empty)
        # Everything that is not the test speaker trains: no clip is dropped.
        self.assertEqual(len(train_df) + len(test_df), len(manifest))
        self.assertTrue(set(train_df["speaker_id"]).isdisjoint(set(test_df["speaker_id"])))

    def test_all_training_data_keeps_more_clips_than_the_carved_split(self) -> None:
        manifest = synthetic_manifest()
        carved, _validation, _test, _ = speaker_disjoint_split(
            manifest, test_speaker="s1", return_validation=True
        )
        everything, _empty, _test2, _ = speaker_disjoint_split(
            manifest,
            test_speaker="s1",
            return_validation=True,
            carve_validation=False,
        )
        self.assertGreater(len(everything), len(carved))

    def test_dataloaders_return_no_validation_loader(self) -> None:
        train_loader, validation_loader, test_loader, label_to_index, _ = (
            get_custom_words_dataloaders(
                batch_size=4,
                speaker_holdout="TestID",
                return_validation=True,
                carve_validation=False,
            )
        )
        self.assertIsNone(validation_loader)
        self.assertEqual(len(train_loader.dataset) + len(test_loader.dataset), 800)
        self.assertEqual(
            set(test_loader.dataset.dataframe["speaker_id"]), {"TestID"}
        )
        self.assertNotIn(
            "TestID", set(train_loader.dataset.dataframe["speaker_id"])
        )
        self.assertEqual(set(label_to_index), set(WORDS))

    def test_carve_validation_without_a_holdout_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a speaker_holdout"):
            get_custom_words_dataloaders(batch_size=4, carve_validation=False)

    def test_default_still_carves_validation(self) -> None:
        """The default must not change: existing callers rely on the 4-tuple."""
        _train, _validation, _test, _ = speaker_disjoint_split(
            synthetic_manifest(), test_speaker="s1", return_validation=True
        )
        self.assertFalse(_validation.empty)


if __name__ == "__main__":
    unittest.main()
