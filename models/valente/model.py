"""PyTorch implementation of the LipNet sentence-level architecture.

The model maps a variable-length mouth-video sequence directly to a sequence of
character distributions. Train its output with :class:`torch.nn.CTCLoss`;
there is deliberately no word-classification or fixed-length spelling head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LipNet(nn.Module):
    """LipNet: STCNN visual frontend, two BiGRUs, and a CTC character head.

    Args:
        num_classes: Number of output symbols, including the CTC blank token.
        input_channels: One for grayscale mouth crops, three for RGB crops.
        hidden_size: Hidden features in each direction of a GRU. The paper
            uses 256.
        dropout: Channel-wise dropout probability used after every pool layer.

    Input is ``[batch, time, channels, height, width]``. ``forward`` returns
    unnormalised logits shaped ``[batch, time, num_classes]``. Call
    ``ctc_log_probs`` to receive the ``[time, batch, classes]`` log-probability
    layout expected by ``torch.nn.CTCLoss``.
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        hidden_size: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include a CTC blank and one symbol.")

        # Paper Appendix A.2: 32, 64, 96 STCNN channels; temporal stride is 1.
        self.stcnn = nn.Sequential(
            self._conv_block(input_channels, 32, dropout),
            self._conv_block(32, 64, dropout),
            self._conv_block(64, 96, dropout),
        )
        self.feature_projection = nn.LazyLinear(hidden_size)
        self.gru1 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.gru2 = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        self._initialise_recurrent_layers()

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(3, 5, 5),
                stride=(1, 2, 2),
                padding=(1, 2, 2),
            ),
            nn.ReLU(inplace=True),
            # Pool spatially only, so CTC receives one emission per input frame.
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(p=dropout),
        )

    def _initialise_recurrent_layers(self) -> None:
        """Use the initialisation described in the LipNet appendix where possible."""
        for module in (self.gru1, self.gru2):
            for name, parameter in module.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(parameter)
                elif "weight_ih" in name:
                    nn.init.kaiming_normal_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
        nn.init.kaiming_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    @staticmethod
    def infer_input_lengths(frames: torch.Tensor) -> torch.Tensor:
        """Infer valid frame counts from zero-padded clips.

        The STCNN never reduces the temporal dimension, so these are also the
        CTC input lengths. Supply explicit lengths to ``forward`` if padding is
        not zero-valued.
        """
        if frames.ndim != 5:
            raise ValueError("frames must have shape [batch, time, channels, height, width].")
        valid = frames.abs().sum(dim=(2, 3, 4)) > 0
        return valid.sum(dim=1).clamp(min=1)

    def forward(
        self,
        frames: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return character logits with shape ``[batch, time, num_classes]``."""
        if frames.ndim != 5:
            raise ValueError("frames must have shape [batch, time, channels, height, width].")
        if lengths is None:
            lengths = self.infer_input_lengths(frames)
        lengths = lengths.to(dtype=torch.long).clamp(min=1, max=frames.size(1))

        # Conv3d uses [batch, channels, time, height, width].
        features = self.stcnn(frames.permute(0, 2, 1, 3, 4))
        batch, channels, time, height, width = features.shape
        features = features.permute(0, 2, 1, 3, 4).contiguous().view(
            batch, time, channels * height * width
        )
        features = self.feature_projection(features)

        packed = nn.utils.rnn.pack_padded_sequence(
            features,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed, _ = self.gru1(packed)
        packed, _ = self.gru2(packed)
        sequence, _ = nn.utils.rnn.pad_packed_sequence(
            packed,
            batch_first=True,
            total_length=time,
        )
        return self.classifier(sequence)

    def ctc_log_probs(
        self,
        frames: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(log_probs, input_lengths)`` in the layout for ``CTCLoss``."""
        if lengths is None:
            lengths = self.infer_input_lengths(frames)
        lengths = lengths.to(dtype=torch.long).clamp(min=1, max=frames.size(1))
        logits = self(frames, lengths)
        return F.log_softmax(logits, dim=-1).transpose(0, 1), lengths


class LipNetCTC(LipNet):
    """Backward-compatible CTC wrapper for the previous Valente notebook.

    ``vocab_size`` excludes the blank token and ``forward`` directly returns
    log-probabilities in PyTorch CTC layout, ``[time, batch, classes]``.
    New code should prefer :class:`LipNet` and ``ctc_log_probs`` explicitly.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        dropout: float = 0.5,
    ) -> None:
        self.vocab_size = vocab_size
        self.blank_index = vocab_size
        super().__init__(
            num_classes=vocab_size + 1,
            input_channels=1,
            hidden_size=hidden_size,
            dropout=dropout,
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        lengths = self.infer_input_lengths(frames)
        logits = super().forward(frames, lengths)
        return F.log_softmax(logits, dim=-1).transpose(0, 1)


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    blank_index: int,
) -> list[list[int]]:
    """Collapse repeated tokens and remove blanks from ``[time, batch, class]``."""
    best_path = log_probs.argmax(dim=-1).transpose(0, 1)
    decoded_sequences: list[list[int]] = []

    for batch_index, sequence in enumerate(best_path):
        collapsed: list[int] = []
        previous: int | None = None
        for token in sequence[: int(input_lengths[batch_index].item())].tolist():
            if token == blank_index:
                previous = None
            elif token != previous:
                collapsed.append(token)
                previous = token
        decoded_sequences.append(collapsed)

    return decoded_sequences


def tokens_to_text(tokens: list[int], num_to_char: dict[int, str]) -> str:
    """Convert decoded token IDs into text with a supplied character mapping."""
    return "".join(num_to_char[token] for token in tokens if token in num_to_char)


# Compatibility name for Valente-specific imports. It is a sequence model,
# not the earlier word-classification recognizer.
ValenteRecognizer = LipNet
