from typing import *

import torch
import torch.nn as nn
from torchvision.models import (
    resnet50, resnet101, ResNet50_Weights, 
    ResNet101_Weights, resnet152, ResNet152_Weights,
)


class ResNet(nn.Module):

    def __init__(
        self, 
        resnet_type: str='resnet50', 
        hidden_dim: int=4096, 
        projection_dim: int=256,
        resnet_last_dim: int=2048,
    ):
        super(ResNet, self).__init__()
        assert resnet_type in ('resnet50', 'resnet101', 'resnet152')

        if resnet_type == 'resnet50':
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif resnet_type == 'resnet101':
            backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        
        self.projector = nn.Sequential(
            nn.Linear(resnet_last_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.projector(x)
        return x