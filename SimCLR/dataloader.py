import random
from PIL import ImageFilter, Image

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def augmentation(size, subset='train'):
    if subset == 'train':
        transformation = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            GaussianBlur(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    elif subset == 'test':
        transformation = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    else:
        raise ValueError(f'{subset} does not exists')

    return transformation


class CIFAR10Pair(dsets.CIFAR10):

    def __getitem__(self, idx):
        images, targets = self.data[idx], self.targets[idx]
        images = Image.fromarray(images)

        if self.transform is not None:
            x_i = self.transform(images)
            x_j = self.transform(images)

        if self.target_transform is not None:
            target = self.target_transform(targets)

        return x_i, x_j, target


def get_dataloader(transforms_, root='cifar10', subset='train',
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