import argparse
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchsummary import summary

from models import StudentNet, TeacherNet
from loss import DistillationLoss
from train import TrainModel
from eval import eval

def get_args_parser():
    parser = argparse.ArgumentParser(description='Knowledge Distillation Baseline Code', add_help=False)
    parser.add_argument('--teacher_lr', default=1e-2, type=float,
                        help='learning rate')
    parser.add_argument('--student_lr', default=1e-2, type=float,
                        help='learning rate')
    parser.add_argument('--teacher_epochs', default=15, type=int,
                        help='Epochs for training model')
    parser.add_argument('--student_epochs', default=15, type=int,
                        help='Epochs for training model')
    parser.add_argument('--teacher_optimizer', default='adam', type=str,
                        help='the optimizer for training teacher network')
    parser.add_argument('--student_optimizer', default='sgd', type=str,
                        help='the optimizer for training student network')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch Size for training model')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='class number of dataset')
    parser.add_argument('--check_point', default=True, type=bool,
                        help='apply check point for saving weights of model')
    parser.add_argument('--train_log_steps', default=40, type=int,
                        help='print log of training phase')
    parser.add_argument('--valid_log_steps', default=20, type=int,
                        help='print log of validating phase')
    return

def main(args):
    train_data = dsets.MNIST(
        root='./minst_data',
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )

    test_data = dsets.MNIST(
        root='./mnist_data',
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )

    train_index, valid_index = train_test_split(
        np.arange(len(train_data)),
        test_size=0.2,
        shuffle=True,
        stratify=train_data.targets,
    )

    # train loader
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(train_index),
        drop_last=True,
    )
    # valid loader
    valid_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(valid_index),
        drop_last=True,
    )
    # test loader
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ############################# Train Teacher Network #############################
    teacher_net = TeacherNet(num_classes=args.num_classes)
    summary(teacher_net, (1, 28, 28), device='cpu')

    teacher_model = TrainModel(
        model=teacher_net,
        lr=args.teacher_lr,
        device=device,
        optimizer=args.teacher_optimizer,
        epochs=args.teacher_epochs,
        phase='teacher',
        train_log_steps=args.train_log_steps,
        valid_log_steps=args.valid_log_steps,
        check_point=args.check_point,
    )

    teacher_history = teacher_model.fit(train_loader, valid_loader)

    ########################## Evaluate Teacher Network ##########################
    eval(model=teacher_history['model'], dataset=test_loader, device=device)

    ##############################################################################



    ########################### Train Student Network ###########################
    student_net = StudentNet(num_classes=args.num_classes)
    summary(student_net, (1, 28, 28), device='cpu')

    distil_loss = DistillationLoss(
        base_criterion=nn.CrossEntropyLoss(),
        teacher_model=teacher_history['model'],
        distillation_type='hard',
        alpha=0.1,
        tau=3,
    )

    student_model = TrainModel(
        model=student_net,
        lr=args.student_lr,
        device=device,
        optimizer=args.student_optimizer,
        loss_func=distil_loss,
        phase='student',
        epochs=args.student_epochs,
        train_log_steps=args.train_log_steps,
        valid_log_steps=args.valid_log_steps,
        check_point=False,
    )

    student_history = student_model.fit(train_loader, valid_loader)

    ########################## Evaluate Student Network ##########################
    eval(model=student_history['model'], dataset=test_loader, device=device)

    ##############################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run Knowledge Distillation Baseline Code', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)