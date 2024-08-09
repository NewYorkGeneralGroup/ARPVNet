import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskAdaptiveLoss(nn.Module):
    def __init__(self, num_classes):
        super(TaskAdaptiveLoss, self).__init__()
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.ones(num_classes))

    def forward(self, pred, target):
        loss = F.cross_entropy(pred, target, reduction='none')
        weighted_loss = loss * self.weight[target]
        return weighted_loss.mean()
