import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self, point_dim=3, voxel_dim=256, num_heads=8):
        super(AttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.point_proj = nn.Linear(point_dim, voxel_dim)
        self.q_proj = nn.Linear(voxel_dim, voxel_dim)
        self.k_proj = nn.Linear(voxel_dim, voxel_dim)
        self.v_proj = nn.Linear(voxel_dim, voxel_dim)
        self.fusion = nn.Linear(voxel_dim * 2, voxel_dim)

    def forward(self, points, voxel_features):
        B, N, _ = points.shape
        point_features = self.point_proj(points)
        
        q = self.q_proj(point_features).view(B, N, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(voxel_features).view(B, -1, self.num_heads, q.size(-1)).transpose(1, 2)
        v = self.v_proj(voxel_features).view(B, -1, self.num_heads, q.size(-1)).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        voxel_features_attended = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, -1)

        fused_features = self.fusion(torch.cat([point_features, voxel_features_attended], dim=-1))
        return fused_features
