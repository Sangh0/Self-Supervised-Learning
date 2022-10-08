import time
import argparse
from tqdm.auto import tqdm

import torch
import torch.optim as optim
from torchsummary import summary

from util.dataset import get_dataset
from util.loss import DINOLoss
from util.util import get_params_groups, cosine_scheduler
from models.model_utils import MultiCropWrapper, DINOHead
from models.resnet import ResNet50
from models import vit
from train import train_on_epoch


def fit(student, teacher, data_loader, dino_loss, epochs,
          optimizer, lr_schedule, wd_schedule, momentum_schedule,
          use_amp, clip_grad, freeze_last_layer, log_step=300):
    
    print('Start Model Training...!')
    start_time = time.time()
    pbar = tqdm(range(epochs), total=int(epochs))
    for epoch in pbar:
        
        init_time = time.time()

        loss = train_on_epoch(
            student=student,
            teacher=teacher,
            data_loader=data_loader,
            dino_loss=dino_loss,
            epoch=epoch,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            momentum_schedule=momentum_schedule,
            use_amp=use_amp,
            clip_grad=clip_grad,
            freeze_last_layer=freeze_last_layer,
            log_step=log_step,
        )

        end_time = time.time()
    
    return


def get_args_parser():
    parser = argparse.ArgumentParser(description='DINO', add_help=False)
    # hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='a batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='initial value of learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='minimum value of learning rate for cosine scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.04,
                        help='initial value of weight decay')
    parser.add_argument('--weight_decay_end', type=float, default=0.4,
                        help='maximum value of weight decay for cosine scheduler')
    parser.add_argument('--warmup_epoch', type=int, default=10,
                        help='number of epochs for warm-up scheduling')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd'],
                        help='optimizer for training model')
    
    # model paramters
    parser.add_argument('--model', type=str, default='vit_t',
                        choices=['resnet', 'vit_t', 'vit_s', 'vit_b'],
                        help='a name of vision transforer or resnet')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='patch size for vision transformer model')
    parser.add_argument('--out_dim', type=int, default=65536,
                        help='dimension of the DINO head output. For complex and large datasets large values work well')
    parser.add_argument('--use_bn_in_head', type=bool, default=False,
                        help='batch normalization in projection head')
    parser.add_argument('--norm_last_layer', type=int, default=True,
                        help='Whether or not to weight normalize the last layer of the DINO head')
    parser.add_argument('--momentum_teacher', type=float, default=0.996,
                        help='Base EMA parameter for teacher update and the value is increased to 1 during training with cosine schedule.')

    # temperature paramters
    parser.add_argument('--warmup_teacher_temp', type=float, default=0.04,
                        help='initial value for the teacher temperature')
    parser.add_argument('--teacher_temp', type=float, default=0.04,
                        help='Final value (after linear warmup) of the teacher temperature')
    parser.add_argument('--warmup_teacher_temp_epochs', type=float, default=0,
                        help='number of warmup epochs for the teacher temperature')

    # augmentation parameters
    parser.add_argument('--global_img_size', type=int, default=224,
                        help='image size of global view in augmentations')
    parser.add_argument('--local_img_size', type=int, default=96, 
                        help='image size of local view in augmentations')
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help='crop scale for global view in augmentations')
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help='crop scale for local view in augmentations')
    parser.add_argument('--local_crops_number', type=float, default=8,
                        help='the number of multi cropping for training')
    

    # etc paramters
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='set device for faster model training')
    return 


def main(args):
    dataset = get_dataset(
        global_img_size=args.global_img_size,
        local_img_size=args.local_img_size,
        global_crops_scale=args.global_crops_scale,
        local_crops_scale=args.local_crops_scale,
        local_crops_number=args.local_crops_number,
        batch_size=args.batch_size,
    )

    train_loader, test_loader = dataset['train'], dataset['test']
    print('data loaders ready...')

    if 'resnet' in args.model:
        student = ResNet50()
        teacher = ResNet50()

    elif 'vit' in args.model:
        if args.model == 'vit_t':
            student = vit.vit_tiny(patch_size=args.patch_size)
            teacher = vit.vit_tiny(patch_size=args.patch_size)

        elif args.model == 'vit_s':
            student = vit.vit_small(patch_size=args.patch_size)
            teacher = vit.vit_small(patch_size=args.patch_size)
        
        elif args.model == 'vit_b':
            student = vit.vit_base(patch_size=args.patch_size)
            teacher = vit.vit_base(patch_size=args.patch_size)
        
        else:
            raise ValueError(f'{args.model} does not exist, choose between vit_t, vit_s or vit_b')

    else:
        raise ValueError(f'{args.model} does not exist, choose vit or resnet')

    embed_dim = student.embed_dim
    student = MultiCropWrapper(
        student,
        DINOHead(
            in_dim=embed_dim,
            out_dim=args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
    )
    teacher = MultiCropWrapper(
        teacher,
        DINOHead(
            in_dim=embed_dim,
            out_dim=args.out_dim,
            use_bn=args.use_bn_in_head,
        )
    )
    print('student and teacher networks ready...')

    dino_loss = DINOLoss(
        out_dim=args.out_dim,
        ncrops=args.local_crops_number + 2,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        epochs=args.epochs,
    )

    params_groups = get_params_groups(student)
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(params_groups)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params_groups)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(params_groups, lr=0, momentum=0.9)
    else:
        raise ValueError(f'{args.optimizer} does not exist')

    lr_schedule = cosine_scheduler(
        base_value=args.lr,
        final_value=args.min_lr,
        epochs=args.epochs,
        niter_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )

    wd_schedule = cosine_scheduler(
        base_value=args.weight_decay,
        final_value=args.weight_decay_end,
        epochs=args.epochs,
        niter_per_epoch=len(train_loader),
    )

    momentum_schedule = cosine_scheduler(
        base_value=args.momentum_teacher,
        final_value=1,
        epochs=args.epochs,
        niter_per_epoch=len(train_loader),
    )
    print('schedulers of learning rate, weight decay and momentum ready')

