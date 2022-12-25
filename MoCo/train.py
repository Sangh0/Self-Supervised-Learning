import argparse
import logging
import time
import os
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from model import ResNet50
from callback import CheckPoint
from dataloader import data_loader


def get_args_parser():
    parser = argparse.ArgumentParser(description='MoCo', add_help=False)
    parser.add_argument('--m', type=int, default=4096,
                        help='the number of negative samples')
    parser.add_argument('--feature_dim', type=int, default=128,
                        help='The dimension of output vector of encoder')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='The temperature value for calculating loss function')
    parser.add_argument('--lr', type=float, default=3e-2,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=200,
                        help='total epoch number for training')
    parser.add_argument('--momentum', type=float, default=0.999,
                        help='The momentum variable for momenutm update')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--check_point', type=bool, default=True,
                        help='save weight if valid loss is lower than previous step loss')
    parser.add_argument('--weight_save_path', type=str, default='./weights',
                        help='a path name of folder for saving weights')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--log_step', type=int, default=30,
                        help='print logs of training loss at each steps')
    return parser


class TrainMoCo(object):

    def __init__(
        self,
        m: int=4096,
        feature_dim: int=128,
        temperature: float=0.07,
        lr: float=3e-2,
        epochs: int=200,
        momentum: float=0.999,
        weight_decay: float=1e-4,
        check_point: bool=True,
        weight_save_path: str='./weights',
        log_step: int=10,
    ):

        self.logger = logging.getLogger('The logs of Learning Model with MoCo')
        self.logger.setLvel(logging.INFO)
        stream_handler = logging.StreamHandler()
        self.logger.addHandler(stream_handler)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f'device is {self.device}')
        
        self.memory_queue = F.normalize(torch.randn(m, feature_dim).to(self.device), dim=-1)
        self.temperature = temperature
        self.momentum = momentum

        self.encoder_q = ResNet50(dim=feature_dim).to(self.device)
        self.encoder_k = ResNet50(dim=feature_dim).to(self.device)
        self.logger.info(f'key encoder and query encoder ready...!')

        # initialize weights
        self.initialize_weights(self.encoder_q, self.encoder_k)

        self.loss_func = nn.CrossEntropyLoss().to(self.device)
        self.logger.info(f'loss function ready...!')

        self.optimizer = optim.SGD(
            self.encoder_q.parameters(), 
            momentum=0.9, 
            lr=lr, 
            weight_decay=weight_decay,
        )
        self.logger.info('optimizer ready...!')

        os.makedirs(weight_save_path, exist_ok=True)
        self.check_point = check_point
        self.cp = CheckPoint(verbose=True)
        self.weight_save_path = weight_save_path

        self.epochs = epochs
        self.log_step = log_step
        self.writer = SummaryWriter()

    def learning_(self, train_loader):
        self.logger.info('\nStart Training...!')
        start_training_time = time.time()
        for epoch in tqdm(range(self.epochs)):
            init_epoch_time = time.time()

            train_loss = self.train_on_batch(train_loader, self.log_step, epoch)
            
            end_epoch_time = time.time()

            self.logger.info(f'\n{" "*30} Epoch {epoch+1}/{self.epochs} {" "*30}'
                             f'\n{" "*15} lr = {self.optimizer.param_groups[0]["lr"]}'
                             f'\n{" "*15} time: {end_epoch_time-init_epoch_time:.3f}s, loss: {train_loss:.3f}')

            self.update_lr(self.optimizer, epoch+1)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)

            if self.check_point:
                path = self.weight_save_path + f'/check_point_{epoch+1}.pt'
                self.cp(train_loss, self.encoder_q, path)

    def train_on_batch(self, data_loader, log_step, epoch):
        self.encoder_q.train()
        total_loss = 0
        for batch, (x_q, x_k, _) in enumerate(data_loader):
            x_q, x_k = x_q.to(self.device), x_k.to(self.device)

            query = self.encoder_q(x_q)

            # shuffle Batch Normalization
            idx = torch.randperm(x_k.size(0), device=x_k.device)
            key = self.encoder_k(x_k[idx])
            key = key[torch.argsort(idx)]

            score_pos = torch.bmm(query.unsqueeze(dim=1), key.unsqueeze(dim=-1)).squeeze(dim=-1)
            score_neg = torch.mm(query, self.memory_queue)
            out = torch.cat([score_pos, score_neg], dim=-1) # shape: [B, 1+M]
            loss = self.loss_func(out / self.temperature, torch.zeros(x_q.size(0)))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # momentum update
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(self.momentum * param_k.data + (1 - self.momentum) * param_q.data)

            # update queue
            self.memory_queue = torch.cat([self.memory_queue, key], dim=0)[key.size(0):]

            total_loss += loss.item()

            if (batch+1) % log_step == 0:
                self.logger.info(f'\n{" "*10} Train Batch {batch+1}/{len(data_loader)} loss: {loss.item():.3f} {" "*10}')

            step = len(data_loader) * epoch + batch
            self.writer.add_scalar('Train/loss', loss.item(), step)

        return total_loss / (batch+1)

    @torch.no_grad()
    def initialize_weights(self, encoder_q, encoder_k):
        for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            # don't update by gradient
            param_k.requires_grad = False

    @torch.no_grad()
    def update_lr(self, optimizer, epoch):
        if 120 <= epoch and epoch < 160:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
        elif 160 <= epoch:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1



def main(args):

    train_transforms_ = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_loader = data_loader(
        transforms_=train_transforms_,
        root='cifar10',
        subset='train',
        batch_size=args.batch_size,
    )

    moco = TrainMoCo(
        m=args.m,
        temperature=args.temperature,
        lr=args.lr,
        epochs=args.epochs,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        check_point=args.check_point,
        weight_save_path=args.weight_save_path,
        log_step=args.log_step,
    )

    moco.learning_(train_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MoCo Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)