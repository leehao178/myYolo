import os
import argparse
import time
import shutil
import datetime
import random
import yaml
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from models.data.datasets.dataset import Dataset
from models.loss.loss import YOLOv3Loss
from models.detector import YOLOv3
from models.eval.evaluator import Evaluator
from loguru import logger
import numpy as np


# python -m torch.distributed.launch --nproc_per_node=3 train.py
# '/home/lab602.demo/.pipeline/datasets/VOCdevkit'
parser = argparse.ArgumentParser(description='Netowks Object Detection Training')
parser.add_argument('-n', '--name', default='voc', type=str, help='name')
parser.add_argument('--data', default='/home/lab602.demo/.pipeline/datasets/VOCdevkit',
                    type=str,
                    metavar='DIR',
                    help='path to dataset')
parser.add_argument('--config_file', default='configs/yolov3.yaml',
                    type=str,
                    help='path to config file')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=14, type=int,
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
parser.add_argument("--pretrained", type=str, default='outputs/darknet53.conv.74',
                    help='node rank for distributed training')

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    init_seeds(0)
    args = parser.parse_args()
    outputs = './outputs/{}'.format(args.name)
    logger.add(f'{outputs}/train_log.txt', encoding='utf-8', enqueue=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

    with open(args.config_file, errors='ignore') as f:
        configs = yaml.safe_load(f)

    milti_gpus = True
    if milti_gpus:
        # 1) ?????????
        torch.distributed.init_process_group(backend="nccl")

        # 2??? ?????????????????????gpu
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    else:
        if args.gpu is not None:
            local_rank = 0
            torch.cuda.set_device(args.gpu)
            device = torch.device('cuda:{}'.format(args.gpu) if True else 'cpu')
            logger.info('Use GPU: {} for training'.format(args.gpu))

    model = YOLOv3(anchors=configs['anchors'],
                   strides=configs['strides'][0],
                   num_classes=configs['nc'],
                   pretrained=args.pretrained)
    model.load_darknet_weights(args.pretrained)
    if milti_gpus:
        # 4) ???????????????????????????????????????gpu
        model.to(device)

        # 5) ??????
        if torch.cuda.device_count() > 1:
            model = nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
    else:
        if args.gpu is not None:
            model.cuda(args.gpu)
    
    # define loss function (criterion) and optimizer
    # criterion = YOLOv3Loss(anchors=ANCHORS, strides=STRIDES,
    #                                 iou_threshold_loss=IOU_THRESHOLD_LOSS)

    if milti_gpus:
        configs['LR'] = configs['LR'] * (3 ** 0.5)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=configs['LR'],
                                momentum=configs['momentum'],
                                weight_decay=float(configs['weight_decay']))
    start_epoch = 0
    # scheduler_LR = StepLR(optimizer, step_size=40, gamma=0.1)
    scheduler_LR = MultiStepLR(optimizer,
                               milestones=[25, 40],
                               gamma=0.1,
                               last_epoch=start_epoch - 1)

    if local_rank == 0:
        tblogger = SummaryWriter(os.path.join(outputs, "tensorboard"))

    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         best_acc1 = checkpoint['best_acc1']
    #         if args.gpu is not None:
    #             # best_acc1 may be from a checkpoint from a different GPU
    #             best_acc1 = best_acc1.to(args.gpu)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    train_sampler = ''
    custom_datasets = Dataset(  img_size=configs['train_size'],
                                batch_size=args.batch_size,
                                workers=args.workers,
                                isDistributed=milti_gpus,
                                data_dir=args.data,
                                dataset_type='voc',
                                configs=configs)
    train_loader, val_loader, train_sampler = custom_datasets.dataloader()

    best_acc1 = 0

    max_iter = len(train_loader)

    n_burnin = min(round(max_iter / 5 + 1), 1000)  # burn-in batches

    criterion = YOLOv3Loss(iou_threshold_loss=0.5)

    for epoch in range(args.start_epoch, args.epochs):
        if milti_gpus:
            # ???????????????????????????????????????
            train_sampler.set_epoch(epoch)
        freeze_backbone = False
        cutoff = 75
        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                # if int(name.split('.')[1]) < cutoff:  # if layer < 75
                #     p.requires_grad = False if epoch == 0 else True
                if name.split('.')[0] == 'backnone':
                    p.requires_grad = False if epoch == 0 else True
        model.train()
        
        # train for one epoch
        losses, xy_loss, wh_loss, obj_loss, cls_loss = train(train_loader,
                                                             model,
                                                             optimizer,
                                                             epoch,
                                                             args,
                                                             local_rank,
                                                             max_iter,
                                                             args.epochs,
                                                             n_burnin, 
                                                             configs,
                                                             criterion)
        if local_rank == 0:
            tblogger.add_scalar("train/loss", losses, epoch + 1)
            tblogger.add_scalar("train/xy_loss", xy_loss, epoch + 1)
            tblogger.add_scalar("train/wh_loss", wh_loss, epoch + 1)
            tblogger.add_scalar("train/obj_loss", obj_loss, epoch + 1)
            tblogger.add_scalar("train/cls_loss", cls_loss, epoch + 1)
            tblogger.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], epoch + 1)
        is_best = False
        if local_rank == 0:
            if (epoch + 1) % 10 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': model.module.state_dict() if type(
                                model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best, outputs=outputs, epoch=epoch + 1)
        if local_rank == 0:
            if (epoch + 1) % 20 == 0:
                mAP = 0
                logger.info('*'*20+"Validate"+'*'*20)
                with torch.no_grad():
                    APs = Evaluator(model).APs_voc()
                    for i in APs:
                        logger.info(f'{i} --> mAP : {APs[i]}')
                        mAP += APs[i]
                    # num_classes = 20
                    mAP = mAP / 20
                    tblogger.add_scalar("val/mAP", mAP, epoch + 1)
                    logger.info(f'mAP:{mAP}')

        scheduler_LR.step()

def save_checkpoint(state, is_best, outputs='', epoch=''):
    filename = os.path.join(outputs, 'epoch_{}.pth'.format(epoch))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(outputs, 'model_best.pth'))

def train(train_loader, model, optimizer, epoch, args, rank, max_iter, max_epoch, n_burnin, configs, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    xy_loss = AverageMeter()
    wh_loss = AverageMeter()
    obj_loss = AverageMeter()
    cls_loss = AverageMeter()

    # switch to train mode
    end = time.time()
    mloss = torch.zeros(4).cuda()  # mean losses
    for i, (imgs, bboxes_xywh, label)  in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        _, _, w, h = imgs.shape

        imgs = imgs.cuda()
        bboxes_xywh = [bbox.cuda() for bbox in bboxes_xywh]
        label = [labels.cuda() for labels in label]

        # SGD burn-in
        if epoch == 0 and i <= n_burnin:
            lr = configs['LR'] * (i / n_burnin) ** 4
            for x in optimizer.param_groups:
                x['lr'] = lr

        # compute output
        p, p_d = model(imgs)
        loss, loss_xywh, loss_conf, loss_cls = criterion(p, p_d, bboxes_xywh, label, model)


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update running mean of tracked metrics
        loss_items = torch.tensor([loss_xywh, loss_conf, loss_cls, loss]).cuda()
        mloss = (mloss * i + loss_items) / (i + 1) # ?????????
        # measure accuracy and record loss
        losses.update(mloss[3].item(), imgs.size(0))
        xy_loss.update(loss_xywh.item(), imgs.size(0))
        wh_loss.update(loss_xywh.item(), imgs.size(0))
        obj_loss.update(loss_conf.item(), imgs.size(0))
        cls_loss.update(loss_cls.item(), imgs.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if rank == 0:
            if i % args.print_freq == 0:
                left_iter = max_iter - i + 1
                eta_seconds = ((max_epoch - epoch + 1) * max_iter *  batch_time.avg) + left_iter *  batch_time.avg
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                    'ETA: {eta}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'img_size: {img_size}\t'
                    'Learing Rate {LR:.4f}\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'xy_loss: {xy_loss.val:.3f}\t'
                    'wh_loss: {wh_loss.val:.3f}\t'
                    'obj_loss: {obj_loss.val:.3f}\t'
                    'cls_loss: {cls_loss.val:.3f}'.format(
                    epoch+1, i, len(train_loader), eta=datetime.timedelta(seconds=int(eta_seconds)), batch_time=batch_time,
                    data_time=data_time, img_size='{}x{}'.format(w, h), loss=losses, LR=optimizer.param_groups[0]['lr'],
                    xy_loss=xy_loss,
                    wh_loss=wh_loss,
                    obj_loss=obj_loss,
                    cls_loss=cls_loss))
    return losses.avg, xy_loss.avg, wh_loss.avg, obj_loss.avg, cls_loss.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()