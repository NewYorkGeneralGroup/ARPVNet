import torch
import torch.nn as nn

class AdaptiveVoxelization(nn.Module):
    def __init__(self, resolutions=[8, 16, 32, 64]):
        super(AdaptiveVoxelization, self).__init__()
        self.resolutions = resolutions

    def forward(self, points, resolution_map):
        # points: (B, N, 3)
        # resolution_map: (B, 4)
        B, N, _ = points.shape
        voxel_features = []

        for i, res in enumerate(self.resolutions):
            voxel_size = 1.0 / res
            voxel_indices = (points / voxel_size).long()
            voxel_features.append(self._voxelize(points, voxel_indices, res) * resolution_map[:, i:i+1])

        return torch.cat(voxel_features, dim=1)

    def _voxelize(self, points, voxel_indices, resolution):
        B, N, _ = points.shape
        voxel_features = torch.zeros(B, resolution**3, 3).to(points.device)
        voxel_indices = voxel_indices[:, :, 0] * resolution**2 + voxel_indices[:, :, 1] * resolution + voxel_indices[:, :, 2]
        voxel_features.scatter_add_(1, voxel_indices.unsqueeze(-1).expand(-1, -1, 3), points)
        return voxel_features
