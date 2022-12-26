import argparse
import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader

from linear_classification import Net


@torch.no_grad()
def eval(model, dataset, loss_func=nn.CrossEntropyLoss()):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    model.eval()
    batch_loss, batch_acc = 0, 0
    pbar = tqdm(enumerate(dataset), total=len(dataset))
    
    start = time.time()
    for batch, (images, labels) in pbar:
        images, labels = images.to(device), labels.to(device)
    
        outputs = model(images)
        loss = loss_func(outputs, labels)
        output_index = torch.argmax(outputs, dim=1)
        acc = (output_index==labels).sum()/len(outputs)

        batch_loss += loss.item()
        batch_acc += acc.item()

    end = time.time()

    print(f'\nTotal time for testing is {end-start:.2f}s')
    print(f'\nAverage loss: {batch_loss/(batch+1):.3f}  accuracy: {batch_acc/(batch+1):.3f}')
    return {
        'loss': batch_loss/(batch+1),
        'accuracy': batch_acc/(batch+1),
    }

def get_args_parser():
    parser = argparse.ArgumentParser(description='Evaluating Our Network', add_help=False)
    parser.add_argument('--pretrained_weight', type=str, required=True,
                        help='load weight file of trained model')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='class number of dataset')
    parser.add_argument('--img_size', type=int, default=224,
                        help='image size used when training')
    return parser


def main(args):

    test_transforms_ = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_data = dsets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transforms_,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        drop_last=True,
    )

    model = Net(num_classes=args.num_classes, pretrained_weight=args.pretrained_weight)

    result = eval(model, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Linear Classification Testing', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)