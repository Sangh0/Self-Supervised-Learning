import torch
import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):
    """ Base Encoder Network """
    def __init__(self, feature_dim: int=128):
        super(ResNet50, self).__init__()

        self.encoder = models.resnet50(weights=None, num_classes=feature_dim)
        mlp_dim = self.encoder.fc.in_features
        
        # base encoder
        self.encoder.fc = nn.Identity()

        # encoder containing projection head
        self.projection = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, feature_dim),
        )

    def forward(self, x_i, x_j):
        out_i = self.encoder(x_i)
        out_j = self.encoder(x_j)

        proj_i = self.projection(out_i)
        proj_j = self.projection(out_j)
        
        return {
            'encoder': [out_i, out_j],
            'projection': [proj_i, proj_j],
        }