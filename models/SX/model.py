from __future__ import annotations

import torch
import torch.nn as nn


class HandSignLSTM(nn.Module):
    """Sequence classifier for MediaPipe hand landmarks.

    Input shape: (batch_size, sequence_length, input_size)
    where input_size is 63 for one-hand data or 126 for two-hand data.
    """

    def __init__(
        self,
        input_size: int = 126,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"Expected input shape (batch, seq_len, features), got {tuple(x.shape)}"
            )
        if x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected feature size {self.input_size}, got {x.size(-1)}"
            )

        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.classifier(out)
