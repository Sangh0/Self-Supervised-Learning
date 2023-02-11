import argparse

from dataloader import augmentation, get_dataloader
from byol import BYOL
from networks import ResNet, MLP


def get_args_parser():
    parser = argparse.ArgumentParser(description='BYOL Training', add_help=False)

    # training parameters
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size for batch training')
    parser.add_argument('--warmup_lr', type=float, default=0.2,
                        help='initial learning rate')
    parser.add_argument('--end_lr', type=float, default=0,
                        help='last learning rate of last epoch for scheduler')
    parser.add_argument('--weight_dcay', type=float, default=1.5e-6,
                        help='weight decay')
    parser.add_argument('--warmup_epoch', type=int, default=10,
                        help='warm up epoch for scheduler')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='epochs for training')
    parser.add_argument('--log_step', type=int, default=30,
                        help='print logs like loss for each iterations')

    # byol hyperparameters
    parser.add_argument('--tau', type=float, default=0.996,
                        help='tau value in exponential-moving average')
    parser.add_argument('--projection_dim', type=int, default=256,
                        help='projection dimension in online network')
    parser.add_argument('--hidden_dim', type=int, default=4096,
                        help='hidden dimension after average pooling in online network')
    
    # data parameters
    parser.add_argument('--img_size', type=int, default=224,
                        help='input image size')

    # model parameters
    parser.add_argument('--resnet_type', type=str, default='resnet50')


def main(args):
    device = torch.device('cuda' if torch.cuda.is_availabel() else 'cpu')

    online_network = ResNet(
        resnet_type=args.resnet_type, 
        hidden_dim=args.hidden_dim, 
        projection_dim=args.projection_dim,
    )

    target_network = online_network

    predictor = MLP(projected_dim=args.projection_dim, hidden_dim=args.hidden_dim)

    augment1, augment2 = augmentation(size=(args.img_size, args.img_size))
    dataloader = get_dataloader(augment1, augment2, batch_size=args.batch_size)

    byol = BYOL(
        device=device,
        warmup_lr=args.warmup_lr,
        end_lr=args.end_lr,
        weight_decay=args.weight_decay,
        warmup_epoch=args.warmup_epoch,
        epochs=args.epochs,
        online_net=online_network,
        target_net=target_network,
        predictor=predictor,
        tau=args.tau,
        log_step=args.log_step,
    )

    # training
    byol._learning(dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BYOL training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)