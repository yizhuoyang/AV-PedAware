import torch
import torch.nn as nn
import torch.nn.functional as F

from network.audio_net import AudioNet


class AudioAngleNet(nn.Module):
    """Predict a categorical azimuth probability distribution from audio only."""

    def __init__(
        self,
        dropout_rate=0.4,
        feature_dim=128,
        hidden_dim=64,
        kernel_num=8,
        audio_channels=4,
        num_angle_bins=360,
    ):
        super().__init__()
        self.num_angle_bins = int(num_angle_bins)
        if self.num_angle_bins <= 1:
            raise ValueError("num_angle_bins must be greater than one")
        self.audio_encoder = AudioNet(
            dropout_rate=dropout_rate,
            feature_dim=feature_dim,
            kerel_num=kernel_num,
            audio_channels=audio_channels,
        )
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, self.num_angle_bins)

    def forward(self, spec):
        feature = self.audio_encoder(spec)
        feature = F.relu(self.fc1(feature))
        feature = self.dropout(feature)
        logits = self.fc2(feature)
        return F.softmax(logits, dim=1)
