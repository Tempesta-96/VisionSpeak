"""LipNet-style 3D-CNN + BiGRU classifier for isolated word lip reading."""

import torch
import torch.nn as nn


class LipReadingClassifier(nn.Module):
    """Classify a mouth-crop sequence into one of ``num_classes`` words.

    Input is ``(B, T, 1, H, W)`` grayscale frames in ``[0, 1]``. A
    spatiotemporal 3D-CNN frontend (LipNet-style) extracts per-timestep
    features, a bidirectional GRU summarises the temporal axis, and a small
    MLP head produces per-word logits.
    """

    def __init__(
        self,
        num_classes: int = 51,
        gru_hidden: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.gru_hidden = gru_hidden

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
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1, H, W) -> (B, 1, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # (B, C, T, h, w)

        batch, channels, time, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, h, w)
        x = x.view(batch, time, -1)  # (B, T, C*h*w)

        _, hidden = self.gru(x)  # hidden: (2, B, H)
        forward_hidden = hidden[0]
        backward_hidden = hidden[1]
        pooled = torch.cat([forward_hidden, backward_hidden], dim=1)  # (B, 2H)

        return self.classifier(pooled)
