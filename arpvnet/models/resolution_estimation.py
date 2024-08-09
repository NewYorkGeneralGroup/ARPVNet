import torch
import torch.nn as nn

class ResolutionEstimationNetwork(nn.Module):
    def __init__(self, input_channels):
        super(ResolutionEstimationNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.fc = nn.Linear(256, 4)  # 4 resolution levels

    def forward(self, x):
        # x: (B, N, C)
        x = x.transpose(1, 2)  # (B, C, N)
        features = self.mlp(x)
        global_features = torch.max(features, dim=2)[0]  # (B, 256)
        resolution_scores = self.fc(global_features)  # (B, 4)
        return torch.softmax(resolution_scores, dim=1)
