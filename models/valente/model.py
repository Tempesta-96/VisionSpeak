"""LipNet-style PyTorch sequence model with CTC decoding helpers."""

from __future__ import annotations

import torch
import torch.nn as nn


class LipNetCTC(nn.Module):
    """A LipNet-inspired 3D CNN + BiLSTM network for character prediction.

    Input shape is ``(B, T, 1, H, W)`` and output is per-timestep
    log-probabilities with shape ``(T, B, C)`` for use with ``nn.CTCLoss``.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.blank_index = vocab_size
        self.num_classes = vocab_size + 1

        self.features = nn.Sequential(
            nn.Conv3d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(256, 75, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
        )
        self.temporal_projection = nn.LazyLinear(hidden_size * 2)
        self.lstm_1 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout_1 = nn.Dropout(dropout)
        self.lstm_2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout_2 = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, self.num_classes)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames.permute(0, 2, 1, 3, 4)
        x = self.features(x)
        batch, channels, time, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(
            batch, time, channels * height * width
        )
        x = self.temporal_projection(x)
        x, _ = self.lstm_1(x)
        x = self.dropout_1(x)
        x, _ = self.lstm_2(x)
        x = self.dropout_2(x)
        logits = self.classifier(x)
        log_probs = logits.log_softmax(dim=-1)
        return log_probs.permute(1, 0, 2)


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    blank_index: int,
) -> list[list[int]]:
    """Greedy CTC decode from ``(T, B, C)`` log-probabilities."""
    best_path = log_probs.argmax(dim=-1).transpose(0, 1)
    decoded_sequences: list[list[int]] = []

    for batch_index, sequence in enumerate(best_path):
        valid_steps = int(input_lengths[batch_index].item())
        collapsed: list[int] = []
        previous = None

        for token in sequence[:valid_steps].tolist():
            if token == blank_index:
                previous = None
                continue
            if token != previous:
                collapsed.append(token)
            previous = token

        decoded_sequences.append(collapsed)

    return decoded_sequences


def tokens_to_text(tokens: list[int], num_to_char: dict[int, str]) -> str:
    return "".join(num_to_char[token] for token in tokens if token in num_to_char)
