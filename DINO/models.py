import timm
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc.out_features = num_classes

    def forward(self, x):
        return self.model(x)


class ViT_S(nn.Module):

    def __init__(self, num_classes, patch_size=8):
        super().__init__()
        assert patch_size in (8, 16), \
            f'patch size {patch_size} does not exist, you have to choose between 8 and 16'

        if patch_size == 8:
            self.model = timm.create_model('vit_small_patch8_224_dino', num_classes=num_classes)
        else:
            self.model = timm.create_model('vit_small_patch16_224_dino', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


class ViT_B(nn.Module):

    def __init__(self, num_classes, patch_size=8):
        super().__init__()
        assert patch_size in (8, 16), \
            f'patch size {patch_size} does not exist, you have to choose between 8 and 16'

        if patch_size == 8:
            self.model = timm.create_model('vit_base_patch8_224_dino', num_classes=num_classes)
        else:
            self.model = timm.create_model('vit_base_patch16_224_dino', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)