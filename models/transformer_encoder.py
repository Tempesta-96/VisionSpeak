"""Temporal transformer building blocks shared by the lip-reading models.

The recurrent cores of ``models/valente/model.py`` and ``models/LBY/model.py``
are replaced by :class:`TemporalTransformerEncoder`. It keeps two things the
GRUs provided and that the rest of the code depends on:

* output shape ``[batch, time, model_dim]``, so the CTC head and the word
  classifier keep working untouched;
* propagates per-sample valid lengths, so padded clips stay masked and
  ``torch.nn.CTCLoss`` still receives truthful input lengths.

Design notes
------------
``norm_first=True`` (pre-layer-norm) is used because it is markedly more stable
when the model is optimised with the small batches these datasets force
(``batch_size=1`` in the custom-words CTC notebook).

Length propagation is supplied by the caller rather than inferred here: after a
strided convolutional frontend the valid length depends on the exact stride
schedule, and guessing it silently would corrupt the CTC loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def strided_length(lengths: torch.Tensor, stride: int) -> torch.Tensor:
    """Return valid lengths after a stride-``stride`` convolution.

    Mirrors ``nn.Conv1d`` semantics with ``padding = stride // 2`` applied to an
    input of ``lengths`` samples: ``floor((L - 1) / stride) + 1``.
    """
    if stride < 1:
        raise ValueError("stride must be at least 1.")
    if stride == 1:
        return lengths.clone()
    return torch.div(lengths - 1, stride, rounding_mode="floor") + 1


def propagate_lengths(lengths: torch.Tensor, strides: tuple[int, ...]) -> torch.Tensor:
    """Apply every stride in ``strides`` in order to per-sample valid lengths."""
    result = lengths.to(dtype=torch.long)
    for stride in strides:
        result = strided_length(result, stride)
    return result.clamp(min=1)


def masked_mean_pool(
    features: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average ``[batch, time, dim]`` over the valid prefix of each sample.

    Padded timesteps never contribute to the result, which matters because the
    custom-words batches are zero-padded to the longest clip.
    """
    if features.ndim != 3:
        raise ValueError("features must have shape [batch, time, dim].")
    if lengths is None:
        return features.mean(dim=1)

    batch, time, _ = features.shape
    positions = torch.arange(time, device=features.device).unsqueeze(0)
    mask = positions < lengths.to(device=features.device).unsqueeze(1)
    weights = mask.unsqueeze(-1).to(dtype=features.dtype)
    total = (features * weights).sum(dim=1)
    count = weights.sum(dim=1).clamp(min=1.0)
    return total / count


class TemporalTransformerEncoder(nn.Module):
    """Stacked pre-layer-norm transformer encoder over a time axis.

    Args:
        model_dim: Width of the token embeddings and of the output features.
        num_layers: Number of encoder layers.
        num_heads: Attention heads; ``model_dim`` must be divisible by this.
        feedforward_dim: Hidden width of each layer's MLP block.
        dropout: Dropout inside the encoder and applied to its output.
        max_tokens: Size of the learned positional embedding table. The
            frontend controls how many tokens a clip produces, so this must be
            comfortably above the largest expected emission count.
        activation: MLP activation inside each encoder layer.

    ``forward`` accepts ``[batch, time, model_dim]`` and returns the same shape.
    """

    def __init__(
        self,
        model_dim: int = 192,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.5,
        max_tokens: int = 96,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if model_dim <= 0:
            raise ValueError("model_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_heads <= 0 or model_dim % num_heads:
            raise ValueError("num_heads must divide model_dim exactly.")
        if feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.model_dim = model_dim
        self.max_tokens = max_tokens

        # Learned absolute positions. The frontend downsamples time, so the
        # token count is roughly T / prod(strides) rather than T.
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, max_tokens, model_dim)
        )
        self.input_dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is only a speed hint for the padded case and
        # conflicts with norm_first; disabling it avoids a spurious warning.
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialise like the reference transformer, matching the paper's scheme."""
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        for name, parameter in self.encoder.named_parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        nn.init.ones_(self.output_norm.weight)
        nn.init.zeros_(self.output_norm.bias)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode ``[batch, time, model_dim]`` features into the same shape.

        ``lengths`` marks the valid prefix of each sample. When supplied it is
        converted into a key-padding mask, which matters more than it might
        look: self-attention mixes every position with every other one, so
        without the mask a zero-padded timestep is indistinguishable from a
        genuinely silent mouth frame and measurably corrupts the valid
        timesteps. A bidirectional GRU does not have this failure mode, so the
        mask is what makes the transformer path behave comparably on the
        variable-length, zero-padded batches this project uses.
        """
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, model_dim].")
        if features.size(-1) != self.model_dim:
            raise ValueError(
                f"Expected feature size {self.model_dim}, got {features.size(-1)}."
            )

        time = features.size(1)
        if time > self.max_tokens:
            raise ValueError(
                f"Sequence length {time} exceeds max_tokens {self.max_tokens}. "
                "Raise max_tokens when constructing the encoder."
            )

        padding_mask: torch.Tensor | None = None
        if lengths is not None:
            lengths = lengths.to(dtype=torch.long)
            if bool((lengths > time).any()):
                raise ValueError("lengths must not exceed the sequence length.")
            positions = torch.arange(time, device=features.device).unsqueeze(0)
            valid = positions < lengths.to(device=features.device).unsqueeze(1)
            # nn.TransformerEncoder uses True to mean "ignore this position".
            padding_mask = ~valid
            # A fully padded row would make every attention weight NaN.
            if bool(padding_mask.all(dim=1).any()):
                raise ValueError("every sample needs at least one valid timestep.")

        hidden = features + self.positional_embedding[:, :time]
        hidden = self.input_dropout(hidden)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return self.output_dropout(self.output_norm(hidden))
