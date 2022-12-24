from PIL import Image

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class CIFAR10Pair(dsets.CIFAR10):

    def __getitem__(self, idx):
        images, targets = self.data[idx], self.targets[idx]
        images = Image.fromarray(images)

        if self.transform is not None:
            pos_1 = self.transform(images)
            pos_2 = self.transform(images)

        if self.target_transform is not None:
            target = self.target_transform(targets)

        return pos_1, pos_2, target


def data_loader(transforms_, root='cifar10', subset='train', 
                download=True, batch_size=32, shuffle=True, drop_last=True):

    subset = True if subset == 'train' else False
    
    dataset = CIFAR10Pair(
        root=root,
        train=subset,
        transform=transforms_,
        download=download,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )

    return dataloader


train_transforms_ = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

test_transforms_ = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])