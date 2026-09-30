"""Transformer sequence classifier for MediaPipe hand/body landmarks.

Drop-in replacement for :class:`models.SX.model.HandSignLSTM`: the same
``(batch, time, input_size) -> (batch, num_classes)`` contract, so
``inference.sx_worker.SXDetector`` and ``models/SX/predict.py`` do not need a
different call shape.

Why a window matters
--------------------
The LSTM this replaces is normally trained with ``sequence_length=1`` and
served one frame at a time, so it never sees motion. Attention over a single
token is degenerate, so this module is only worth using with a real temporal
window (see ``models/SX/train.py``, ``--window-size``). A ``time`` of 1 is still
accepted for backward compatibility.

Pooling
-------
A learnable ``[CLS]`` token is prepended and its final state is classified,
which is the standard formulation for sequence classification. ``"mean"``
pooling over the valid timesteps is also available for short windows, where a
dedicated summary token has little to attend over.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.transformer_encoder import masked_mean_pool


class GestureTransformer(nn.Module):
    """Sequence classifier over landmark vectors.

    Args:
        input_size: Features per frame: 63 for one hand, 126 for two hands,
            225 when two hands and body pose are both enabled.
        num_classes: Number of sign/gesture classes.
        d_model: Token width. ``input_size`` is projected up to this.
        num_layers: Transformer encoder layers.
        num_heads: Attention heads; ``d_model`` must be divisible by this.
        feedforward_dim: MLP width inside each encoder layer.
        dropout: Dropout for the encoder, the pooled feature, and the head.
        max_tokens: Positional embedding table size. It must cover
            ``window_size + 1`` to allow for the prepended ``[CLS]`` token.
        pooling: ``"cls"`` or ``"mean"``.

    ``forward`` takes ``(batch, time, input_size)`` and returns unnormalised
    logits shaped ``(batch, num_classes)``. Passing ``lengths`` marks the valid
    prefix of each window so padded timesteps are excluded.
    """

    def __init__(
        self,
        input_size: int = 126,
        num_classes: int = 10,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.2,
        max_tokens: int = 33,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_heads <= 0 or d_model % num_heads:
            raise ValueError(
                f"num_heads must divide d_model exactly, got num_heads={num_heads} "
                f"and d_model={d_model}."
            )
        if feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive.")
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}.")
        if max_tokens < 2:
            raise ValueError("max_tokens must be at least 2 to fit the CLS token.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_size = input_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.pooling = pooling
        self.max_tokens = max_tokens

        self.input_projection = nn.Linear(input_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        self.input_dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        for name, parameter in self.encoder.named_parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        nn.init.ones_(self.output_norm.weight)
        nn.init.zeros_(self.output_norm.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"Expected input shape (batch, seq_len, features), got {tuple(x.shape)}"
            )
        if x.size(-1) != self.input_size:
            raise ValueError(f"Expected feature size {self.input_size}, got {x.size(-1)}")

        batch, time, _ = x.shape
        total_tokens = time + 1  # + [CLS]
        if total_tokens > self.max_tokens:
            raise ValueError(
                f"Window of {time} frames needs {total_tokens} token positions but "
                f"max_tokens is {self.max_tokens}."
            )

        hidden = self.input_projection(x)
        hidden = torch.cat([self.cls_token.expand(batch, -1, -1), hidden], dim=1)
        hidden = hidden + self.positional_embedding[:, :total_tokens]
        hidden = self.input_dropout(hidden)

        padding_mask = None
        if lengths is not None:
            lengths = lengths.to(dtype=torch.long, device=x.device)
            if bool((lengths > time).any()):
                raise ValueError("lengths must not exceed the sequence length.")
            # The CLS token is always valid; real frames are valid up to lengths.
            positions = torch.arange(time, device=x.device).unsqueeze(0)
            frame_valid = positions < lengths.unsqueeze(1)
            cls_valid = torch.ones(batch, 1, dtype=torch.bool, device=x.device)
            padding_mask = ~torch.cat([cls_valid, frame_valid], dim=1)

        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = self.output_dropout(self.output_norm(hidden))

        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            # Exclude the CLS token from the mean over real frames.
            frame_lengths = (
                lengths if lengths is not None else torch.full((batch,), time, device=x.device)
            )
            pooled = masked_mean_pool(hidden[:, 1:], frame_lengths)

        return self.classifier(pooled)


def build_gesture_model(settings: dict, num_classes: int) -> nn.Module:
    """Construct the gesture model a checkpoint declares.

    Checkpoints written before the transformer experiment have no
    ``architecture`` key and therefore load as ``"lstm"``.
    """
    if not isinstance(settings, dict):
        raise TypeError("settings must be a mapping of checkpoint values.")

    from models.SX.model import HandSignLSTM

    architecture = str(settings.get("architecture", "lstm")).lower()
    input_size = int(settings["input_size"])

    if architecture == "lstm":
        return HandSignLSTM(
            input_size=input_size,
            hidden_size=int(settings.get("hidden_size", 64)),
            num_layers=int(settings.get("num_layers", 2)),
            num_classes=num_classes,
            dropout=float(settings.get("dropout", 0.2)),
        )
    if architecture == "transformer":
        return GestureTransformer(
            input_size=input_size,
            num_classes=num_classes,
            d_model=int(settings.get("d_model", 128)),
            num_layers=int(settings.get("num_layers", 2)),
            num_heads=int(settings.get("num_heads", 4)),
            feedforward_dim=int(settings.get("feedforward_dim", 256)),
            dropout=float(settings.get("dropout", 0.2)),
            max_tokens=int(settings.get("max_tokens", 33)),
            pooling=str(settings.get("pooling", "cls")),
        )
    raise ValueError(f"architecture must be 'lstm' or 'transformer', got {architecture!r}.")
