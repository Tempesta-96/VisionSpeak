"""PyTorch implementations of the LipNet-style sequence models.

The model maps a variable-length mouth-video sequence directly to a sequence of
character distributions. Train its output with :class:`torch.nn.CTCLoss`;
there is deliberately no word-classification or fixed-length spelling head.

Two temporal cores are available through the ``architecture`` argument:

``"bigru"``
    The original LipNet design: two bidirectional GRUs. This remains the
    default so checkpoints trained before the transformer experiment keep
    loading unchanged.
``"transformer"``
    The same STCNN visual frontend followed by a pre-layer-norm transformer
    encoder. Only the recurrent core is replaced, so accuracy differences can
    be attributed to attention-versus-recurrence rather than to a redesigned
    convolutional frontend.

CTC length budget
-----------------
CTC requires ``input_length >= 2 * target_length + 1`` (one emission per
character plus a separator blank). The longest word in the custom dataset is
``thanks`` (6 characters), so a clip needs at least 13 valid emissions. The
transformer path strides time by 2 in the first convolution only, which keeps
the shortest recorded clip (45 frames) at 23 emissions. Do not increase the
temporal stride without re-checking that inequality.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_encoder import (
    TemporalTransformerEncoder,
    propagate_lengths,
)

Architecture = Literal["bigru", "transformer"]

#: Temporal stride applied by each STCNN block on the transformer path.
#: The first block halves the frame rate; the rest preserve it so CTC still
#: receives enough emissions for the longest word (see module docstring).
TRANSFORMER_TEMPORAL_STRIDES: tuple[int, ...] = (2, 1, 1)


class LipNet(nn.Module):
    """LipNet: STCNN visual frontend, temporal core, and a CTC character head.

    Args:
        num_classes: Number of output symbols, including the CTC blank token.
        input_channels: One for grayscale mouth crops, three for RGB crops.
        hidden_size: Hidden features in each direction of a GRU (``"bigru"``)
            or the transformer width (``"transformer"``). The paper uses 256.
        dropout: Channel-wise dropout probability used after every pool layer.
        architecture: ``"bigru"`` for the original two bidirectional GRUs,
            ``"transformer"`` for a transformer encoder.
        transformer_layers: Encoder layers on the transformer path.
        transformer_heads: Attention heads on the transformer path.
        transformer_feedforward: MLP width inside each encoder layer.
        transformer_max_tokens: Size of the learned positional embedding table.

    Input is ``[batch, time, channels, height, width]``. ``forward`` returns
    unnormalised logits shaped ``[batch, time, num_classes]``. Call
    ``ctc_log_probs`` to receive the ``[time, batch, classes]`` log-probability
    layout expected by ``torch.nn.CTCLoss``.

    Note:
        ``feature_projection`` is an ``nn.LazyLinear``, so parameters are only
        materialised after one forward pass. Run a dummy forward before reading
        ``state_dict`` or counting parameters.
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        hidden_size: int = 256,
        dropout: float = 0.5,
        architecture: Architecture = "bigru",
        transformer_layers: int = 3,
        transformer_heads: int = 4,
        transformer_feedforward: int = 512,
        transformer_max_tokens: int = 96,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include a CTC blank and one symbol.")
        if architecture not in ("bigru", "transformer"):
            raise ValueError(
                f"architecture must be 'bigru' or 'transformer', got {architecture!r}."
            )

        self.architecture = architecture
        self.hidden_size = hidden_size
        self.temporal_strides = (
            TRANSFORMER_TEMPORAL_STRIDES if architecture == "transformer" else (1, 1, 1)
        )

        # Paper Appendix A.2: 32, 64, 96 STCNN channels; temporal stride is 1.
        # On the transformer path the first block strides time by 2 so that
        # self-attention cost and activation memory stay linear-ish in length.
        self.stcnn = nn.Sequential(
            self._conv_block(input_channels, 32, dropout, self.temporal_strides[0]),
            self._conv_block(32, 64, dropout, self.temporal_strides[1]),
            self._conv_block(64, 96, dropout, self.temporal_strides[2]),
        )
        self.feature_projection = nn.LazyLinear(hidden_size)

        if architecture == "bigru":
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
            head_input = hidden_size * 2
        else:
            self.encoder = TemporalTransformerEncoder(
                model_dim=hidden_size,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                feedforward_dim=transformer_feedforward,
                dropout=dropout,
                max_tokens=transformer_max_tokens,
            )
            head_input = hidden_size

        self.classifier = nn.Linear(head_input, num_classes)
        self._initialise_parameters()

    @staticmethod
    def _conv_block(
        in_channels: int,
        out_channels: int,
        dropout: float,
        temporal_stride: int = 1,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(3, 5, 5),
                stride=(temporal_stride, 2, 2),
                padding=(1, 2, 2),
            ),
            nn.ReLU(inplace=True),
            # Pool spatially only, so CTC receives one emission per surviving
            # timestep.
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(p=dropout),
        )

    def _initialise_parameters(self) -> None:
        """Use the initialisation described in the LipNet appendix where possible."""
        if self.architecture == "bigru":
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

    def emission_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """Map frame counts to the emission counts this architecture produces.

        The BiGRU path never reduces time; the transformer path applies
        :attr:`temporal_strides` in the STCNN, so CTC must be given the
        shortened lengths or its loss silently becomes wrong.
        """
        return propagate_lengths(lengths, self.temporal_strides)

    @staticmethod
    def infer_input_lengths(frames: torch.Tensor) -> torch.Tensor:
        """Infer valid frame counts from zero-padded clips.

        Supply explicit lengths to ``forward`` if padding is not zero-valued.
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
        """Return character logits with shape ``[batch, time, num_classes]``.

        On the transformer path the time axis is shortened by
        :attr:`temporal_strides`; use :meth:`emission_lengths` to obtain the
        matching CTC input lengths.
        """
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

        emission_lengths = self.emission_lengths(lengths).clamp(max=time)

        if self.architecture == "bigru":
            packed = nn.utils.rnn.pack_padded_sequence(
                features,
                emission_lengths.cpu(),
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
        else:
            sequence = self.encoder(features, emission_lengths)

        return self.classifier(sequence)

    def ctc_log_probs(
        self,
        frames: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(log_probs, input_lengths)`` in the layout for ``CTCLoss``.

        ``input_lengths`` are emission counts, not frame counts, so they stay
        correct on architectures that downsample time.
        """
        if lengths is None:
            lengths = self.infer_input_lengths(frames)
        lengths = lengths.to(dtype=torch.long).clamp(min=1, max=frames.size(1))
        logits = self(frames, lengths)
        emission_lengths = self.emission_lengths(lengths).clamp(max=logits.size(1))
        return F.log_softmax(logits, dim=-1).transpose(0, 1), emission_lengths


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
        architecture: Architecture = "bigru",
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.blank_index = vocab_size
        super().__init__(
            num_classes=vocab_size + 1,
            input_channels=1,
            hidden_size=hidden_size,
            dropout=dropout,
            architecture=architecture,
            **kwargs,
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        lengths = self.infer_input_lengths(frames)
        logits = super().forward(frames, lengths)
        return F.log_softmax(logits, dim=-1).transpose(0, 1)


def build_lipnet(
    settings: dict,
    num_classes: int,
    *,
    input_channels: int = 1,
) -> LipNet:
    """Construct a :class:`LipNet` from a checkpoint's saved settings.

    Checkpoints written before the transformer experiment have no
    ``architecture`` key and therefore load as ``"bigru"``. Unknown keys are
    tolerated so a checkpoint can carry extra metadata such as the training
    configuration without breaking inference.
    """
    if not isinstance(settings, dict):
        raise TypeError("settings must be a mapping of checkpoint values.")

    architecture = str(settings.get("architecture", "bigru")).lower()
    known = {
        "hidden_size",
        "dropout",
        "transformer_layers",
        "transformer_heads",
        "transformer_feedforward",
        "transformer_max_tokens",
    }
    options = {key: settings[key] for key in known if key in settings}
    return LipNet(
        num_classes=num_classes,
        input_channels=input_channels,
        architecture=architecture,  # type: ignore[arg-type]
        **options,
    )


def architecture_from_checkpoint(checkpoint: dict) -> str:
    """Return the architecture name stored in a checkpoint, defaulting to BiGRU."""
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a mapping.")
    return str(checkpoint.get("architecture", "bigru")).lower()


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
