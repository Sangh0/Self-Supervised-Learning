import random
from typing import *

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def augmentation(size: Tuple[int]=(224,224), mode=):
    transformation = [
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                p=0.8
            ),
    ]

    augment1 = transformation.append(
        transforms.GaussianBlur(p=1.0),
        transforms.RandomSolarize(p=0.0),
    )

    augment2 = transformation.append(
        transforms.GaussianBlur(p=0.1),
        transforms.RandomSolarize(p=0.2),
    )

    augment1.append(
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )

    augment2.append(
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )

    return augment1, augment2


def get_dataloader(augment1, augment2, batch_size: int=32, subset: str='train'):
    subset = True if subset == 'train' else False

    dataset1 = dsets.CIFAR10(
        root='./data',
        train=subset,
        download=True,
        transform=augment1,
    )

    dataloader1 = DataLoader(
        dataset1,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    dataset2 = dsets.CIFAR10(
        root='./data',
        train=subset,
        download=True,
        transform=augment2,
    )

    dataloader2 = DataLoader(
        dataset2,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return dataloader1, dataloader2