import os
import argparse
import time
import shutil
import cv2
from cv2 import imwrite
import torch
from torch import nn
from models.heads.classifier import Classifier
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import datetime
from models.data.datasets.dataset import Dataset
from models.loss.loss import compute_loss
from models.detector import YOLOv3
from models.eval.evaluator import Evaluator
from models.eval.evaluator import non_max_suppression
from models.data.datasets.voc import resize
import torchvision.transforms as transforms
from models.utils.visualization import _COLORS
import numpy as np


parser = argparse.ArgumentParser(description='Netowks Object Detection Training')
parser.add_argument('-n', '--name', default='voc', type=str, help='name')
parser.add_argument('--data', default='/home/lab602.demo/.pipeline/datasets/VOCdevkit',
                    type=str,
                    metavar='DIR',
                    help='path to dataset')
parser.add_argument('--img_size', default=416,
                    type=int,
                    help='path to dataset')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument("--local_rank", type=int, default=0,
                    help='node rank for distributed training')

parser.add_argument("-c", "--cpkt", type=str, default='/home/lab602.demo/.pipeline/10678031/myYolo/outputs/voc/epoch_50.pth',
                    help='pth')


def main():
    conf_thres = 0.5
    nms_thres = 0.5
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda:{}'.format(args.gpu) if True else 'cpu')
    print("Use GPU: {} for training".format(args.gpu))

    classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    ANCHORS = [[(10,13), (16,30), (33,23)],  # Anchors for small obj
                [(30,61, ), (62,45), (59,119)],  # Anchors for medium obj
                [(116,90), (156,198), (373,326)]]# Anchors for big obj
    STRIDES = [8, 16, 32]
    ANCHORS_PER_SCLAE = 3

    model = YOLOv3(anchors=torch.FloatTensor(ANCHORS).to(device),
                   strides=torch.FloatTensor(STRIDES).to(device))
    
    model.cuda(args.gpu)
    # print(args.cpkt)
    model.load_state_dict(torch.load(args.cpkt, map_location=device)['model'])
    path = '/home/lab602.demo/.pipeline/10678031/myYolo/dog.jpg'
    img = cv2.imread(path)
    img = transforms.ToTensor()(img)
    img = resize(img, 416).unsqueeze(0).to(device)
    # print(img.shape)
    # img = torch.from_numpy(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        pred, _ = model(img)
    print(pred.shape)
    det = non_max_suppression(pred, conf_thres, nms_thres)[0]

    img = cv2.imread(path)
    w, h, _ = img.shape
    img = cv2.resize(img, (416, 416))
    
    for i in det:
        xmin, ymin, xmax, ymax, conf, score2, cls = i


        color = (_COLORS[int(cls.cpu())] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(classes[int(cls)], conf * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[int(cls.cpu())]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        txt_bk_color = (_COLORS[int(cls.cpu())] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (int(xmin), int(ymin) + 1),
            (int(xmin) + txt_size[0] + 1, int(ymin) + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (int(xmin), int(ymin) + txt_size[1]), font, 0.4, txt_color, thickness=1)
    img = cv2.resize(img, (h, w))
    cv2.imwrite('/home/lab602.demo/.pipeline/10678031/myYolo/dog_test.jpg', img)


if __name__ == '__main__':
    main()