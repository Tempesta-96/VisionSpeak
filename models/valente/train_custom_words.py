"""Train the CTC (LipNet) lip reader on the custom isolated-word dataset.

This is the command-line counterpart of ``custom_word_training.ipynb``. The
notebook is fine for exploration but it splits clips at random, which puts the
same signer in train and test, and it selects its checkpoint on that same set.
This script reuses the repository's speaker-disjoint protocol instead, so its
numbers mean what they claim:

* ``--speaker-holdout SPEAKER`` keeps that speaker's clips out of training
  entirely and reports on them at the end.
* ``--all-training-data`` additionally drops the validation carve-out, so every
  remaining clip trains and the final epoch is the deliverable. The test
  speaker is then not read until training is over, which makes the reported
  accuracy a held-out number rather than a selected one.

The temporal core is selectable with ``--architecture``; both options share the
STCNN frontend, so a difference between them is attention versus recurrence:

* ``bigru`` (default): the original two bidirectional GRUs.
* ``transformer``: a pre-layer-norm transformer encoder.

Every architecture setting is written into the checkpoint, so
``inference.lip_worker`` rebuilds the right model without being told which one
it is looking at.

Two accuracies are reported on the test speaker, and they are not the same
number:

* **greedy** — CTC greedy decode, counted correct only when the decoded string
  is exactly the target word. A near-miss like ``helo`` scores zero.
* **scored** — the deployed path (:func:`inference.lip_worker.score_vocabulary`),
  which ranks the eight known words by CTC sequence likelihood. This is what
  the webcam tools and ``compare_lip_models.py`` use, so it is the number to
  quote when comparing against the results in the README.
"""

from __future__ import annotations
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.prepare_custom_word_dataset import (
    available_speakers,
    choose_holdout_speaker,
    load_manifest,
    speaker_disjoint_split,
    stratified_split,
)
from inference.lip_worker import WORDS, score_vocabulary
from models.valente.model import build_lipnet, ctc_greedy_decode, tokens_to_text

EPOCHS = 200
LR = 1e-4
MIN_LR = 1e-6
BATCH_SIZE = 16
GRAD_CLIP = 5.0
SEED = 42
CHECKPOINT = Path("checkpoints/valente_custom_words_lipnet_ctc.pt")
#: Index of the CTC blank. Character indices therefore start at 1.
CTC_BLANK = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--architecture",
        choices=("bigru", "transformer"),
        default="bigru",
        help="Temporal core to train (default: bigru, the original LipNet core).",
    )
    parser.add_argument("--layers", type=int, default=3, help="Transformer encoder layers.")
    parser.add_argument("--heads", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument("--feedforward", type=int, default=512, help="Encoder MLP width.")
    parser.add_argument("--hidden", type=int, default=256, help="GRU or transformer width.")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument(
        "--min-lr",
        type=float,
        default=MIN_LR,
        help="Floor of the cosine schedule (default: 1e-6).",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=GRAD_CLIP,
        help="Max gradient norm; CTC on a 3D-CNN benefits from clipping (default: 5.0).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-dir", type=Path, default=Path("data/custom_words"))
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda (ROCm reports as cuda) or mps.",
    )
    parser.add_argument(
        "--speaker-holdout",
        nargs="?",
        const="auto",
        default=None,
        metavar="SPEAKER",
        help=(
            "Evaluate speaker-disjoint: hold out one speaker entirely. Pass a "
            "speaker id, or use the flag alone to auto-pick the most "
            "representative one. Without it the default per-clip split leaks "
            "speakers between train and test."
        ),
    )
    parser.add_argument(
        "--all-training-data",
        action="store_true",
        help=(
            "Train on every clip from the non-test speakers and keep the final "
            "epoch, instead of carving a validation split out of them. The test "
            "speaker is not evaluated until training ends, so nothing selects on "
            "it. Requires --speaker-holdout to name that test speaker."
        ),
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    """Resolve ``auto``/``cuda``/``mps``/``cpu`` to a concrete device."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "cuda/ROCm was requested but torch sees no device. On ROCm, check "
            "that /dev/kfd and /dev/dri/renderD128 are visible to this process."
        )
    return torch.device(name)


def build_char_vocab(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Map characters to ``1..N``; index 0 is reserved for the CTC blank."""
    characters = sorted(set("".join(sorted(set(words)))))
    char_to_index = {character: index + 1 for index, character in enumerate(characters)}
    index_to_char = {index: character for character, index in char_to_index.items()}
    return char_to_index, index_to_char


class WordClipCTCDataset(Dataset):
    """One clip per item: standardized mouth frames plus its characters.

    Clips are preloaded as uint8 by :func:`preload_clips`, so this only pays for
    the float conversion and per-clip standardization. Padding stays zero, which
    is what ``LipNet.infer_input_lengths`` expects if lengths are ever omitted.
    """

    def __init__(
        self,
        features: np.ndarray,
        frame_lengths: np.ndarray,
        words: np.ndarray,
        char_to_index: dict[str, int],
    ) -> None:
        self.features = features
        self.frame_lengths = frame_lengths
        self.words = words
        self.char_to_index = char_to_index

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        length = int(self.frame_lengths[index])
        real_frames = self.features[index, :length].astype(np.float32) / 255.0
        # Same per-clip standardisation as the word classifier's dataset.
        real_frames = (real_frames - real_frames.mean()) / (real_frames.std() + 1e-6)
        clip = np.zeros_like(self.features[index], dtype=np.float32)
        clip[:length] = real_frames
        word = str(self.words[index])
        target = torch.tensor(
            [self.char_to_index[character] for character in word], dtype=torch.long
        )
        return torch.from_numpy(clip).unsqueeze(1), target, length, word


def preload_clips(dataframe) -> tuple[np.ndarray, np.ndarray]:
    """Read every ``.npz`` into one ``(N, T_max, H, W)`` uint8 array."""
    clips: list[np.ndarray] = []
    frame_shape: tuple[int, ...] | None = None
    for row in dataframe.itertuples(index=False):
        with np.load(Path(row.clip_path), allow_pickle=False) as archive:
            clip = np.asarray(archive["frames"]).copy()
        if frame_shape is None:
            frame_shape = clip.shape[1:]
        elif clip.shape[1:] != frame_shape:
            raise ValueError(f"Inconsistent frame size in {Path(row.clip_path).name}.")
        clips.append(clip)

    if not clips:
        raise ValueError("No clips to preload.")
    frame_lengths = np.array([clip.shape[0] for clip in clips], dtype=np.int64)
    features = np.zeros((len(clips), int(frame_lengths.max()), *frame_shape), dtype=np.uint8)
    for index, clip in enumerate(clips):
        features[index, : clip.shape[0]] = clip
    return features, frame_lengths


def ctc_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, int, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Pad clips to the batch maximum and flatten the character targets."""
    batch = sorted(batch, key=lambda item: item[0].shape[0], reverse=True)
    clips, targets, lengths, words = zip(*batch)
    max_time = max(clip.shape[0] for clip in clips)
    padded = torch.zeros(len(clips), max_time, *clips[0].shape[1:], dtype=clips[0].dtype)
    for index, clip in enumerate(clips):
        padded[index, : clip.shape[0]] = clip
    return (
        padded,
        torch.cat(targets),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor([len(target) for target in targets], dtype=torch.long),
        list(words),
    )


def decode_batch(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    index_to_char: dict[int, str],
) -> list[str]:
    tokens = ctc_greedy_decode(log_probs.detach().cpu(), input_lengths.cpu(), CTC_BLANK)
    return [tokens_to_text(sequence, index_to_char) for sequence in tokens]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    index_to_char: dict[int, str],
    *,
    training: bool,
    grad_clip: float,
) -> tuple[float, float]:
    """One pass, returning ``(mean loss, greedy exact-match accuracy)``."""
    model.train(training)
    total_loss = 0.0
    exact = 0
    total = 0
    for clips, targets, lengths, target_lengths, words in loader:
        clips, targets = clips.to(device), targets.to(device)
        with torch.set_grad_enabled(training):
            log_probs, output_lengths = model.ctc_log_probs(clips, lengths)
            loss = criterion(log_probs, targets, output_lengths.cpu(), target_lengths.cpu())
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        predictions = decode_batch(log_probs, output_lengths, index_to_char)
        exact += sum(prediction == word for prediction, word in zip(predictions, words))
        total += len(words)
        total_loss += float(loss.item()) * len(words)
    return total_loss / max(total, 1), exact / max(total, 1)


@torch.inference_mode()
def evaluate_deployed(
    model: nn.Module,
    dataframe,
    device: torch.device,
    char_to_index: dict[str, int],
) -> tuple[list[str], list[str], list[float]]:
    """Score every clip through the deployed vocabulary-ranking path.

    Batch size is one because :func:`score_vocabulary` ranks the whole word list
    for a single clip, which is exactly what
    :class:`inference.lip_worker.LipPredictor` does at inference time.
    """
    model.eval()
    features, frame_lengths = preload_clips(dataframe)
    labels = dataframe["label"].astype(str).to_numpy()
    truth: list[str] = []
    predicted: list[str] = []
    confidences: list[float] = []

    for index in range(len(labels)):
        length = int(frame_lengths[index])
        clip = features[index, :length].astype(np.float32) / 255.0
        clip = (clip - clip.mean()) / (clip.std() + 1e-6)
        padded = np.zeros_like(features[index], dtype=np.float32)
        padded[:length] = clip
        tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(2).to(device)
        lengths = torch.tensor([length], dtype=torch.long)
        log_probs, output_lengths = model.ctc_log_probs(tensor, lengths)
        scores = score_vocabulary(
            log_probs, output_lengths, WORDS, char_to_index, blank_index=CTC_BLANK
        )
        best = max(scores, key=scores.get)
        truth.append(str(labels[index]))
        predicted.append(best)
        confidences.append(float(scores[best]))

    return truth, predicted, confidences


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)

    manifest = load_manifest(args.data_dir)
    holdout = args.speaker_holdout
    if holdout == "auto":
        holdout = choose_holdout_speaker(manifest)
    elif holdout is not None:
        known = available_speakers(manifest)
        if holdout not in known:
            raise SystemExit(f"Unknown speaker {holdout!r}. Available: {known}.")

    if args.all_training_data and holdout is None:
        raise SystemExit(
            "--all-training-data needs --speaker-holdout SPEAKER to name the test "
            "speaker kept out of training."
        )

    if holdout is None:
        train_df, test_df = stratified_split(manifest, test_size=0.2, seed=args.seed)
        validation_df = None
        print(
            "Split: per-clip random. Every speaker appears in BOTH train and test, "
            "so this accuracy is a same-speaker upper bound. Re-run with "
            "--speaker-holdout for a meaningful number."
        )
    else:
        if args.all_training_data:
            train_df, validation_df, test_df, holdout = speaker_disjoint_split(
                manifest,
                test_speaker=holdout,
                seed=args.seed,
                return_validation=True,
                carve_validation=False,
            )
        else:
            train_df, validation_df, test_df, holdout = speaker_disjoint_split(
                manifest,
                test_speaker=holdout,
                seed=args.seed,
                return_validation=True,
            )

    char_to_index, index_to_char = build_char_vocab(
        list(train_df["label"].astype(str)) or list(manifest["label"].astype(str))
    )
    num_classes = len(char_to_index) + 1

    # The deployed scorer ranks every word in WORDS, so a character that never
    # appeared in training would make it raise deep inside score_vocabulary.
    # Fail here instead, while the cause is still obvious.
    unencodable = {
        word: sorted(set(word).difference(char_to_index))
        for word in WORDS
        if set(word).difference(char_to_index)
    }
    if unencodable:
        raise SystemExit(
            "The training split cannot encode every vocabulary word, so the "
            f"deployed scorer could not rank them: {unencodable}."
        )

    settings = {
        "architecture": args.architecture,
        "hidden_size": args.hidden,
        "dropout": args.dropout,
        "transformer_layers": args.layers,
        "transformer_heads": args.heads,
        "transformer_feedforward": args.feedforward,
    }

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Architecture: {args.architecture}")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    if validation_df is not None and len(validation_df) and not args.all_training_data:
        print(f"Validation samples: {len(validation_df)} (non-test speakers)")
    if holdout is not None:
        print(f"Split: speaker-disjoint, holding out {holdout!r} entirely.")
        if args.all_training_data:
            print("Mode: ALL training data; final epoch is the deliverable.")
    print(f"Character vocabulary ({len(char_to_index)}): {sorted(char_to_index)}")

    train_features, train_lengths = preload_clips(train_df)
    train_words = train_df["label"].astype(str).to_numpy()
    train_loader = DataLoader(
        WordClipCTCDataset(train_features, train_lengths, train_words, char_to_index),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ctc_collate,
        drop_last=False,
    )

    validation_loader = None
    if validation_df is not None and len(validation_df) and not args.all_training_data:
        validation_features, validation_lengths = preload_clips(validation_df)
        validation_loader = DataLoader(
            WordClipCTCDataset(
                validation_features,
                validation_lengths,
                validation_df["label"].astype(str).to_numpy(),
                char_to_index,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=ctc_collate,
        )

    model = build_lipnet(settings, num_classes=num_classes, input_channels=1).to(device)
    # feature_projection is an nn.LazyLinear, so run one batch through before
    # building the optimizer: that materialises the parameter list and makes the
    # reported parameter count real rather than zero.
    with torch.no_grad():
        probe_length = int(train_lengths[0])
        probe = train_features[0, :probe_length].astype(np.float32) / 255.0
        probe = (probe - probe.mean()) / (probe.std() + 1e-6)
        probe = torch.from_numpy(probe).unsqueeze(0).unsqueeze(2).to(device)
        model.ctc_log_probs(probe, torch.tensor([probe_length], dtype=torch.long))
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # Built once, outside the epoch loop: preloading the test clips re-reads
    # every .npz, which is pure waste when it happens once per epoch.
    leaky_test_loader = None if holdout is not None else _test_loader(
        test_df, char_to_index, args.batch_size
    )
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            index_to_char,
            training=True,
            grad_clip=args.grad_clip,
        )
        scheduler.step()
        line = (
            f"Epoch {epoch:3d}/{args.epochs} | train_loss {train_loss:.4f} "
            f"train_greedy {train_accuracy:.4f} | lr {optimizer.param_groups[0]['lr']:.1e}"
        )
        if args.all_training_data:
            print(line, flush=True)
            continue

        selection_loader = validation_loader if validation_loader is not None else leaky_test_loader
        selection_loss, selection_accuracy = run_epoch(
            model,
            selection_loader,
            criterion,
            None,
            device,
            index_to_char,
            training=False,
            grad_clip=args.grad_clip,
        )
        selection_name = "valid" if validation_loader is not None else "test"
        print(
            f"{line} | {selection_name}_loss {selection_loss:.4f} "
            f"{selection_name}_greedy {selection_accuracy:.4f}",
            flush=True,
        )
        if selection_loss < best_loss:
            best_loss = selection_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "char_to_index": char_to_index,
                    "blank_index": CTC_BLANK,
                    "validation_loss": selection_loss,
                    "selection_split": selection_name,
                    **settings,
                },
                args.checkpoint,
            )

    if args.all_training_data:
        torch.save(
            {
                "epoch": args.epochs,
                "model_state_dict": model.state_dict(),
                "char_to_index": char_to_index,
                "blank_index": CTC_BLANK,
                "validation_loss": None,
                "selection_split": "final-epoch-no-validation",
                "trained_on": "all non-test-speaker clips",
                "test_speaker": holdout,
                **settings,
            },
            args.checkpoint,
        )
        print(
            f"Saved final-epoch checkpoint (epoch {args.epochs}, {len(train_df)} "
            f"training clips) to {args.checkpoint}"
        )
    else:
        print(f"Best {best_loss:.4f} selection loss -> {args.checkpoint}")

    # ---- Final evaluation on the test speaker ----
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = _test_loader(test_df, char_to_index, args.batch_size)
    _, greedy_accuracy = run_epoch(
        model,
        test_loader,
        criterion,
        None,
        device,
        index_to_char,
        training=False,
        grad_clip=args.grad_clip,
    )
    print(f"\nGreedy exact-match test accuracy: {greedy_accuracy:.4f}")

    truth, predicted, confidences = evaluate_deployed(model, test_df, device, char_to_index)
    overall = float(np.mean([t == p for t, p in zip(truth, predicted)]))
    print(f"Deployed vocabulary-scored test accuracy: {overall:.4f}")
    print(f"Mean confidence on the chosen word: {float(np.mean(confidences)):.4f}")

    print("\nPer-word accuracy (deployed scoring):")
    for word in WORDS:
        mask = [index for index, value in enumerate(truth) if value == word]
        if not mask:
            continue
        hits = sum(1 for index in mask if predicted[index] == word)
        print(f"  {word:>8}: {hits / len(mask):.4f} ({len(mask)} samples)")

    print("\nConfusion matrix (rows=truth, cols=prediction):")
    matrix = np.zeros((len(WORDS), len(WORDS)), dtype=int)
    word_index = {word: index for index, word in enumerate(WORDS)}
    for target, guess in zip(truth, predicted):
        matrix[word_index[target], word_index[guess]] += 1
    print("        " + " ".join(f"{word[:4]:>4}" for word in WORDS))
    for index, word in enumerate(WORDS):
        print(f"{word:>8} " + " ".join(f"{value:4d}" for value in matrix[index]))


def _test_loader(
    test_df,
    char_to_index: dict[str, int],
    batch_size: int,
) -> DataLoader:
    features, frame_lengths = preload_clips(test_df)
    return DataLoader(
        WordClipCTCDataset(
            features, frame_lengths, test_df["label"].astype(str).to_numpy(), char_to_index
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ctc_collate,
    )


if __name__ == "__main__":
    main()
