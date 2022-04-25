import os
import argparse
import time
import shutil
import cv2
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import datetime
from models.data.datasets.dataset import Dataset


from models.detector import YOLOv3
from models.eval.evaluator import Evaluator
# python -m torch.distributed.launch --nproc_per_node=3 train.py
# '/home/lab602.demo/.pipeline/datasets/VOCdevkit'
parser = argparse.ArgumentParser(description='Netowks Object Detection Training')
parser.add_argument('-n', '--name', default='voc', type=str, help='name')
parser.add_argument('--data', default='/home/lab602.demo/.pipeline/datasets/VOCdevkit',
                    type=str,
                    metavar='DIR',
                    help='path to dataset')
parser.add_argument('--img_size', default=544,
                    type=int,
                    help='path to dataset')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=2, type=int,
                    help='GPU id to use.')
parser.add_argument("--local_rank", type=int, default=0,
                    help='node rank for distributed training')
parser.add_argument("-c", "--cpkt", type=str, default='/home/lab602.demo/.pipeline/10678031/myYolo/outputs/voc/epoch_50.pth',
                    help='pth')





def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda:{}'.format(args.gpu) if True else 'cpu')
    print("Use GPU: {} for training".format(args.gpu))

    ANCHORS = [[(10,13), (16,30), (33,23)],  # Anchors for small obj
                    [(30,61, ), (62,45), (59,119)],  # Anchors for medium obj
                    [(116,90), (156,198), (373,326)]]# Anchors for big obj
    STRIDES = [8, 16, 32]
    ANCHORS_PER_SCLAE = 3

    model = YOLOv3(anchors=torch.FloatTensor(ANCHORS).to(device),
                    strides=torch.FloatTensor(STRIDES).to(device))
    
    model.cuda(args.gpu)
    model.load_state_dict(torch.load(args.cpkt, map_location=device)['model'])
    custom_datasets = Dataset(img_size=args.img_size,
                        batch_size=args.batch_size,
                        workers=args.workers,
                        isDistributed=False,
                        data_dir=args.data,
                        dataset_type='voc')
    train_loader, val_loader, train_sampler = custom_datasets.dataloader()

    mAP = 0
    print('*'*20+"Validate"+'*'*20)
    with torch.no_grad():
        APs = Evaluator(model,
                        dataloader=val_loader,
                        test_size=args.img_size,
                        conf_thres=0.01,
                        nms_thresh=0.5).APs_voc()
        for i in APs:
            print("{} --> mAP : {}".format(i, APs[i]))
            mAP += APs[i]
        # num_classes = 20
        mAP = mAP / 20
        print('mAP:%g'%(mAP))


if __name__ == '__main__':
    main()