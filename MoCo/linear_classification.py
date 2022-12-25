import os
import logging
import argparse
import time
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchsummary import summary

from model import ResNet50
from dataloader import data_loader
from callback import CheckPoint, EarlyStopping


def get_args_parser():
    parser = argparse.ArgumentParser(description='MoCo Linear Classification', add_help=False)
    parser.add_argument('--num_classes', type=int, required=True,
                        help='the number of classes')
    parser.add_argument('--lr', type=float, default=3e-2,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=200,
                        help='total epoch number for training')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--check_point', type=bool, default=True,
                        help='save weight if valid loss is lower than previous step loss')
    parser.add_argument('--early_stop', type=bool, default=False,
                        help='save weight if valid loss is largger than previous step loss with early stopping')
    parser.add_argument('--pretrained_weight_path', type=str, required=True,
                        help='the weight')
    parser.add_argument('--weight_save_path', type=str, default='./weights',
                        help='a path name of folder for saving weights')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--train_log_step', type=int, default=30,
                        help='print logs of training loss at each steps')
    parser.add_argument('--valid_log_step', type=int, default=10,
                        help='print logs of validating loss at each steps')
    return parser


class Net(nn.Module):
    
    def __init__(self, num_classes, pretrained_weight):
        super(Net, self).__init__()
        # encoder
        self.encoder = ResNet50()
        self.encoder.load_state_dict(torch.load(pretrained_weight, map_location='cpu'), strict=False)
        del self.encoder.fc

        # classifier
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

class TrainModel(object):

    def __init__(
        self,
        num_classes,
        pretrained_weight_path,
        lr: float=1e-3,
        epochs: int=100,
        weight_decay: float=1e-6,
        check_point: bool=True,
        early_stop: bool=False,
        weight_save_path: str='./weights',
        train_log_step=30,
        valid_log_step=10,
    ):

        self.logger = logging.getLogger('The logs of Learning Model with MoCo')
        self.logger.setLvel(logging.INFO)
        stream_handler = logging.StreamHandler()
        self.logger.addHandler(stream_handler)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = Net(num_classes=num_classes, pretrained_weight=pretrained_weight_path)
        summary(self.model, (3, 224, 224), device='cpu')
        self.model = self.model.to(self.device)

        # freeze weights for training linear classification network
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        self.logger.info('We freeze the weights of model')

        self.loss_func = nn.CrossEntropyLoss().to(self.device)
        self.logger.info('loss function ready...1')

        self.epochs = epochs

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.logger.info('optimizer ready...!')

        os.makkedirs(weight_save_path, exist_ok=True)
        self.check_point = check_point
        self.cp = CheckPoint(verbose=True)

        self.early_stop = early_stop
        self.es = EarlyStopping(verbose=True, patience=15, path=weight_save_path)

        self.train_log_step = train_log_step
        self.valid_log_step = valid_log_step

    def fit(self, train_data, validation_data):
        print('Start Model Training...!')
        loss_list, acc_list = [], []
        val_loss_list, val_acc_list = [], []
        start_training = time.time()
        pbar = tqdm(range(self.epochs), total=int(self.epochs))
        for epoch in pbar:
            init_time = time.time()
            
            ######### Train Phase #########
            train_loss, train_acc = self.train_on_batch(
                train_data, self.train_log_step,
            )

            loss_list.append(train_loss)
            acc_list.append(train_acc)
            
            ######### Validate Phase #########
            valid_loss, valid_acc = self.validate_on_batch(
                validation_data, self.valid_log_step,
            )
            
            val_loss_list.append(valid_loss)
            val_acc_list.append(valid_acc)

            end_time = time.time()
            
            print(f'\n{"="*30} Epoch {epoch+1}/{self.epochs} {"="*30}'
                  f'\ntime: {end_time-init_time:.2f}s'
                  f'   lr = {self.optimizer.param_groups[0]["lr"]}')
            print(f'\ntrain average loss: {train_loss:.3f}'
                  f'  accuracy: {train_acc:.3f}')
            print(f'\nvalid average loss: {valid_loss:.3f}'
                  f'  accuracy: {valid_acc:.3f}')
            print(f'\n{"="*80}')

            if self.check_point:
                path = f'./weights/check_point_{epoch+1}.pt'
                self.cp(valid_loss, self.model, path)

            if self.early_stop:
                self.es(valid_loss, self.model)
                if self.es.early_stop:
                    print('\n##########################\n'
                          '##### Early Stopping #####\n'
                          '##########################')
                    break

        end_training = time.time()
        print(f'\nTotal time for training is {end_training-start_training:.2}s')
        
        return {
            'model': self.model,
            'loss': loss_list,
            'acc': acc_list,
            'val_loss': val_loss_list,
            'val_acc': val_acc_list,
        }
        
    @torch.no_grad()
    def validate_on_batch(self, validation_data, log_step):
        self.model.eval()
        batch_loss, batch_acc = 0, 0
        for batch, (images, labels) in enumerate(validation_data):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.loss_func(outputs, labels)
            output_index = torch.argmax(outputs, dim=1)
            acc = (output_index==labels).sum()/len(outputs)
            
            if (batch+1) % log_step == 0:
                print(f'\n[Batch {batch+1}/{len(validation_data)}]'
                      f'  valid loss: {loss:.3f}  accuracy: {acc:.3f}')
                
            batch_loss += loss.item()
            batch_acc += acc.item()
            
        return batch_loss/(batch+1), batch_acc/(batch+1)
        
    def train_on_batch(self, train_data, log_step):
        self.model.train()
        batch_loss, batch_acc = 0, 0
        for batch, (images, labels) in enumerate(train_data):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.loss_func(outputs, labels)
            output_index = torch.argmax(outputs, dim=1)
            acc = (output_index==labels).sum()/len(outputs)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                
            if (batch+1) % log_step == 0:
                print(f'\n[Batch {batch+1}/{len(train_data)}]'
                      f'  train loss: {loss:.3f}  accuracy: {acc:.3f}')
                
            batch_loss += loss.item()
            batch_acc += acc.item()
        
        return batch_loss/(batch+1), batch_acc/(batch+1)


def main(args):

    train_transforms_ = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_data = dsets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transforms_
    )

    # stratify split
    train_index, valid_index = train_test_split(
        np.arange(len(train_data)),
        test_size=0.2,
        shuffle=True,
        stratify=train_data.targets,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(train_index),
        drop_last=True,
    )

    valid_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(train_index),
        drop_last=True,
    )

    model = TrainModel(
        num_classes=args.num_classes,
        pretrained_weight_path=args.pretrained_weight_path,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        check_point=args.check_point,
        early_stop=args.early_stop,
        weight_save_path=args.weight_save_path,
        train_log_step=args.train_log_step,
        valid_log_step=args.valid_log_step,
    )

    history = model.fit(train_loader, valid_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Linear Classification Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)