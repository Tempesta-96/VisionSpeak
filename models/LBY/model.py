"""LipNet-style 3D-CNN + temporal-core classifier for isolated word lip reading.

The temporal core is selectable so the recurrent and attention variants can be
compared without touching the convolutional frontend:

``"bigru"``
    The original single bidirectional GRU, reading the last hidden state.
    Kept as the default so existing checkpoints load unchanged.
``"transformer"``
    A pre-layer-norm transformer encoder over the per-timestep features. A
    transformer has no final hidden state to read, so the sequence is reduced
    with attention pooling (:class:`AttentionPooling`) instead.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from models.transformer_encoder import TemporalTransformerEncoder, masked_mean_pool

Architecture = Literal["bigru", "transformer"]

#: Selectable non-linearity between the head's two linear layers. Neither
#: option has parameters, so a checkpoint is agnostic to which one built it —
#: only the recorded setting decides how it is rebuilt. See the class docstring
#: for why ``"relu"`` remains the default but is not what new runs should use.
HEAD_ACTIVATIONS: dict[str, type[nn.Module]] = {"relu": nn.ReLU, "gelu": nn.GELU}


class AttentionPooling(nn.Module):
    """Reduce ``[batch, time, dim]`` to ``[batch, dim]`` with learned attention.

    Padded timesteps are masked to ``-inf`` before the softmax so they can never
    attract attention weight.
    """

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(model_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, dim].")

        logits = self.score(features).squeeze(-1)  # [batch, time]
        if lengths is not None:
            time = features.size(1)
            positions = torch.arange(time, device=features.device).unsqueeze(0)
            valid = positions < lengths.to(device=features.device).unsqueeze(1)
            logits = logits.masked_fill(~valid, float("-inf"))

        weights = torch.softmax(logits, dim=1)
        # Guard against a fully masked row producing NaN weights.
        weights = torch.nan_to_num(weights, nan=0.0)
        return torch.einsum("bt,btc->bc", weights, features)


class LipReadingClassifier(nn.Module):
    """Classify a mouth-crop sequence into one of ``num_classes`` words.

    Input is ``(B, T, 1, H, W)`` grayscale frames in ``[0, 1]``. A
    spatiotemporal 3D-CNN frontend (LipNet-style) extracts per-timestep
    features, a temporal core summarises the sequence, and a small MLP head
    produces per-word logits.

    Args:
        num_classes: Number of isolated words.
        gru_hidden: Width of the BiGRU, or the transformer width when
            ``architecture="transformer"``.
        dropout: Dropout used in the head and, on the transformer path, inside
            the encoder.
        architecture: ``"bigru"`` or ``"transformer"``.
        pooling: Sequence reduction on the transformer path. ``"attention"``
            uses :class:`AttentionPooling`; ``"mean"`` uses masked mean
            pooling.
        head_activation: ``"gelu"`` or ``"relu"``, the non-linearity between the
            head's two linear layers.

    Note:
        For the default 50x100 input, the spatial output is 96 x 3 x 6, so the
        first linear layer sees ``feature_dim = 96 * 3 * 6``. Unlike
        ``models.valente.model.LipNet`` this frontend is not lazy, so the input
        geometry is fixed at construction time.

    Note:
        ``head_activation`` defaults to ``"relu"`` only so that the word
        checkpoints trained before this option existed rebuild exactly as they
        were trained. Prefer ``"gelu"`` for new runs, and note that
        ``models/LBY/train_custom_words.py`` records it as ``"gelu"``.

        The default is not a recommendation. A ReLU whose input goes negative
        receives exactly zero gradient, so once the BiGRU path drove all 256
        head units negative the layer could never recover — and because
        backprop through it was zero, the GRU and the 3D-CNN froze with it. The
        head then returned a constant (its bias), and every clip was labelled
        with whichever class had the largest bias. GELU has a non-zero gradient
        for negative inputs, so that failure cannot be permanent.
    """

    def __init__(
        self,
        num_classes: int = 51,
        gru_hidden: int = 256,
        dropout: float = 0.5,
        architecture: Architecture = "bigru",
        pooling: str = "attention",
        transformer_layers: int = 3,
        transformer_heads: int = 4,
        transformer_feedforward: int = 512,
        transformer_max_tokens: int = 96,
        head_activation: str = "relu",
    ) -> None:
        super().__init__()
        if architecture not in ("bigru", "transformer"):
            raise ValueError(
                f"architecture must be 'bigru' or 'transformer', got {architecture!r}."
            )
        if pooling not in ("attention", "mean"):
            raise ValueError(f"pooling must be 'attention' or 'mean', got {pooling!r}.")
        if head_activation not in HEAD_ACTIVATIONS:
            raise ValueError(
                f"head_activation must be one of {sorted(HEAD_ACTIVATIONS)}, "
                f"got {head_activation!r}."
            )

        self.architecture = architecture
        self.pooling = pooling
        self.gru_hidden = gru_hidden
        self.head_activation = head_activation

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        # For the default 50x100 input, the spatial output is 96 x 3 x 6.
        self.feature_dim = 96 * 3 * 6

        if architecture == "bigru":
            self.gru = nn.GRU(
                input_size=self.feature_dim,
                hidden_size=gru_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            head_input = gru_hidden * 2
        else:
            self.input_projection = nn.Linear(self.feature_dim, gru_hidden)
            self.encoder = TemporalTransformerEncoder(
                model_dim=gru_hidden,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                feedforward_dim=transformer_feedforward,
                dropout=dropout,
                max_tokens=transformer_max_tokens,
            )
            self.attention_pool = (
                AttentionPooling(gru_hidden) if pooling == "attention" else None
            )
            head_input = gru_hidden

        self.classifier = nn.Sequential(
            nn.Linear(head_input, 256),
            HEAD_ACTIVATIONS[head_activation](),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-word logits shaped ``(B, num_classes)``.

        ``lengths`` optionally marks the valid prefix of each clip; padded
        timesteps are excluded from the transformer path's pooling. The BiGRU
        path ignores it, matching the original behaviour.
        """
        if x.ndim != 5:
            raise ValueError("x must have shape (B, T, 1, H, W).")

        # x: (B, T, 1, H, W) -> (B, 1, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # (B, C, T, h, w)

        batch, channels, time, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, h, w)
        x = x.view(batch, time, -1)  # (B, T, C*h*w)

        if self.architecture == "bigru":
            _, hidden = self.gru(x)  # hidden: (2, B, H)
            forward_hidden = hidden[0]
            backward_hidden = hidden[1]
            pooled = torch.cat([forward_hidden, backward_hidden], dim=1)  # (B, 2H)
        else:
            features = self.input_projection(x)
            features = self.encoder(features, lengths)
            if self.attention_pool is not None:
                pooled = self.attention_pool(features, lengths)
            else:
                pooled = masked_mean_pool(features, lengths)

        return self.classifier(pooled)


def build_lip_reading_classifier(settings: dict, num_classes: int) -> LipReadingClassifier:
    """Construct a classifier from checkpoint settings.

    Checkpoints written before the transformer experiment carry no
    ``architecture`` key and therefore load as ``"bigru"``. Likewise, checkpoints
    written before the head non-linearity became selectable carry no
    ``head_activation`` key and rebuild with the ReLU they were trained with, so
    their predictions are unchanged.
    """
    if not isinstance(settings, dict):
        raise TypeError("settings must be a mapping of checkpoint values.")

    known = {
        "gru_hidden",
        "dropout",
        "pooling",
        "transformer_layers",
        "transformer_heads",
        "transformer_feedforward",
        "transformer_max_tokens",
        "head_activation",
    }
    options = {key: settings[key] for key in known if key in settings}
    return LipReadingClassifier(
        num_classes=num_classes,
        architecture=str(settings.get("architecture", "bigru")).lower(),  # type: ignore[arg-type]
        **options,
    )
