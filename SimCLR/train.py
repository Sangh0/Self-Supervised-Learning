import argparse
import logging
import time
import os
from tqdm.auto import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from model import ResNet50
from loss import NTXentLoss
from optimizer import LARS
from callback import CheckPoint
from scheduler import LinearWarmupAndCosineAnneal
from dataloader import augmentation, get_dataloader


def get_args_parser():
    parser = argparse.ArgumentParser(description='SimCLR', add_help=False)

    # SimCLR parameters
    parser.add_argument('--feature_dim', type=int, default=128,
                        help='The dimension of output vector of encoder')
    parser.add_argument('--temperature', type=float, default=0.05,
                        help='The temperature value for calculating loss function')
    parser.add_argument('--use_cosine_similarity', type=bool, default=True,
                        help='using cosine similarity for calculating NT-Xent loss')

    # image size parameters
    parser.add_argument('--img_size', type=int, default=224,
                        help='image size')

    # hyperparameters for training
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='total epoch number for training')
    parser.add_argument('--warmup_epoch', type=int, default=10,
                        help='warm-up epoch for scheduling')
    parser.add_argument('--lr_scheduling', type=bool, default=True,
                        help='apply learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
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


class TrainSimCLR(object):

    def __init__(
        self,
        feature_dim=128,
        lr: float=1e-3,
        weight_decay: float=1e-5,
        epochs: int=300,
        batch_size: int=32,
        temperature: float=0.05,
        use_cosine_similarity: bool=True,
        check_point: bool=True,
        warm_up: int=10,
        lr_scheduling: bool=True,
        weight_save_path: str='./weights',
        log_step: int=10,
    ):

        self.logger = logging.getLogger('The logs of Training Model with SimCLR')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())

        self.device = torch.device('cuda' if torch.cuda.is_availabel() else 'cpu')
        self.logger.info(f'device is {self.device}')

        self.model = ResNet50(feature_dim=feature_dim).to(self.device)

        self.loss_func = NTXentLoss(
            batch_size, 
            temperature=temperature, 
            use_cosine_similarity=use_cosine_similarity,
        )
        self.logger.info('NT-Xent loss function ready...!')

        self.optimizer = LARS(
            self.model.parameters(), 
            lr=lr,
            weight_decay=weight_decay,
        )
        self.logger.info('LARS optimizer ready...!')

        t_max = epochs // warm_up // 2
        self.lr_scheduling = lr_scheduling
        self.scheduler = LinearWarmupAndCosineAnneal(
            optimizer=self.optimizer, 
            warm_up=warm_up,
            T_max=t_max,
        )

        os.makedirs(weight_save_path, exist_ok=True)
        self.check_point = check_point
        self.cp = CheckPoint(verbose=True)
        self.weight_save_path = weight_save_path

        self.epochs = epochs
        self.log_step = log_step
        self.writer = SummaryWriter()

    def learning_(self, train_loader):
        self.logger.info('\nStart Training...!')
        start_train_time = time.time()

        for epoch in tqdm(range(self.epochs)):
            init_epoch_time = time.time()

            train_loss = self.train_on_batch(train_loader, self.log_step, epoch)

            end_epoch_time = time.time()

            self.logger.info(f'\n{" "*30} Epoch {epoch+1}/{self.epochs} {" "*30}'
                             f'\n{" "*15} lr = {self.optimizer.param_groups[0]["lr"]}'
                             f'\n{" "*15} time: {end_epoch_time-init_epoch_time:.3f}s, loss: {train_loss:.3f}')
            
            if self.lr_scheduling:
                self.scheduler.step()

            if self.check_point:
                path = self.weight_save_path + f'/check_point_{epoch+1}.pt'
                self.cp(train_loss, self.model, path)

    def train_on_batch(self, data_loader, log_step, epoch):
        self.model.train()
        total_loss = 0
        for batch, (x_i, x_j, _) in enumerate(data_loader):
            x_i = x_i.to(self.device)
            x_j = x_j.to(self.device)

            outputs = self.model(x_i, x_j)
            out_i, out_j = outputs['encoder'][0], outputs['encoder'][1]
            loss = self.loss_func(out_i, out_j)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (batch+1) % log_step == 0:
                self.logger.info(f'\n{" "*10} Train Batch {batch+1}/{len(data_loader)} loss: {loss.item():.3f} {" "*10}')

            step = len(data_loader) * epoch + batch
            self.writer.add_scalar('Train/loss', loss.item(), step)

        return total_loss / (batch+1)


def main(args):

    transform_ = augmentation(size=args.img_size, subset='train')
    train_loader = get_dataloader(transforms_=transform_, batch_size=args.batch_size)

    simclr = TrainSimCLR(
        feature_dim=args.feature_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        use_cosine_similarity=args.use_cosine_similarity,
        check_point=args.check_point,
        warm_up=args.warmup_epoch,
        lr_scheduling=args.lr_scheduling,
        weight_save_path=args.weight_save_path,
        log_step=args.log_step,
    )

    simclr.learning_(train_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SimCLR Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)