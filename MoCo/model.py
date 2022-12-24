import torch
import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):

    def __init__(self, dim=128):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(2048, dim)

    def forward(self, x):
        out = self.resnet(x)
        norm = torch.norm(out, p='fro', dim=1, keepdim=True)
        return out / norm
    