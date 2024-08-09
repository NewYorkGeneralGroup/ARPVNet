import torch
import torch.nn as nn

class HierarchicalFeatureExtraction(nn.Module):
    def __init__(self, input_dim=3, output_dim=256):
        super(HierarchicalFeatureExtraction, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        self.conv_out = nn.Conv3d(256, output_dim, kernel_size=1)

    def forward(self, x):
        # x: list of 4 tensors with shapes (B, 3, 8^3), (B, 3, 16^3), (B, 3, 32^3), (B, 3, 64^3)
        features = []
        for i, voxel_feature in enumerate(x):
            B, C, N = voxel_feature.shape
            size = int(N**(1/3))
            voxel_feature = voxel_feature.view(B, C, size, size, size)
            
            if i == 0:
                feature = self.conv1(voxel_feature)
            elif i == 1:
                feature = self.conv2(feature) + self.conv1(voxel_feature)
            elif i == 2:
                feature = self.conv3(feature) + self.conv2(voxel_feature)
            else:
                feature = self.conv3(feature) + self.conv3(voxel_feature)
            
            features.append(feature)

        output = self.conv_out(features[-1])
        return output.view(B, -1, N)
