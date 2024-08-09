import torch
import torch.nn as nn
from .resolution_estimation import ResolutionEstimationNetwork
from ..layers.adaptive_voxelization import AdaptiveVoxelization
from ..layers.attention_fusion import AttentionFusion
from ..layers.hierarchical_features import HierarchicalFeatureExtraction

class ARPVNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, num_points=10000):
        super(ARPVNet, self).__init__()
        self.num_points = num_points
        self.ren = ResolutionEstimationNetwork(input_channels)
        self.adaptive_voxelization = AdaptiveVoxelization()
        self.hierarchical_features = HierarchicalFeatureExtraction()
        self.attention_fusion = AttentionFusion()
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, N, 3)
        resolution_map = self.ren(x)
        voxel_features = self.adaptive_voxelization(x, resolution_map)
        hierarchical_features = self.hierarchical_features(voxel_features)
        point_features = self.attention_fusion(x, hierarchical_features)
        logits = self.classifier(point_features)
        return logits
