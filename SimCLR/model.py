import torch
import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):
    """ Base Encoder Network """
    def __init__(self, feature_dim: int=128, proj: bool=False):
        super(ResNet50, self).__init__()

        self.proj = proj

        self.encoder = models.resnet50(weights=None, num_classes=feature_dim)
        mlp_dim = self.encoder.fc.in_features
        
        # base encoder
        self.encoder.fc = nn.Identity()

        # add projection head
        self.projection = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, feature_dim),
        )

    def forward(self, x):
        out = self.encoder(x)

        if not self.proj:
            proj = self.projection(out)
        
            return out, proj

        else:
            return out