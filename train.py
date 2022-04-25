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
from models.loss.loss import compute_loss
from models.detector import YOLOv3
from models.eval.evaluator import Evaluator
from loguru import logger


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
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-b', '--batch-size', default=8, type=int,
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
parser.add_argument("--pretrained", type=str, default='outputs/yolov3.weights',
                    help='node rank for distributed training')


def main():
    args = parser.parse_args()
    logger.add('outputs/voc/train_log.txt', encoding='utf-8', enqueue=True)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    milti_gpus = True
    if milti_gpus:
        # 1) 初始化
        torch.distributed.init_process_group(backend="nccl")

        # 2） 配置每個進程的gpu
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    else:
        if args.gpu is not None:
            local_rank = 0
            torch.cuda.set_device(args.gpu)
            device = torch.device('cuda:{}'.format(args.gpu) if True else 'cpu')
            logger.info('Use GPU: {} for training'.format(args.gpu))


    ANCHORS = [[(10,13), (16,30), (33,23)],  # Anchors for small obj
                [(30,61, ), (62,45), (59,119)],  # Anchors for medium obj
                [(116,90), (156,198), (373,326)]]# Anchors for big obj
    STRIDES = [8, 16, 32]
    ANCHORS_PER_SCLAE = 3

    model = YOLOv3(anchors=torch.FloatTensor(ANCHORS).to(device),
                   strides=torch.FloatTensor(STRIDES).to(device),
                   num_classes=20,
                   pretrained=args.pretrained)

    if milti_gpus:
        # 4) 封裝之前要把模型移到對應的gpu
        model.to(device)

        # 5) 封裝
        if torch.cuda.device_count() > 1:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
    else:
        if args.gpu is not None:
            model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    # criterion = YoloV3Loss(anchors=ANCHORS, strides=STRIDES,
    #                                 iou_threshold_loss=IOU_THRESHOLD_LOSS)
    if milti_gpus:
        args.lr = args.lr * (3 ** 0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # scheduler_LR = StepLR(optimizer, step_size=40, gamma=0.1)
    scheduler_LR = MultiStepLR(optimizer, milestones=[47, 49], gamma=0.1)
    
    outputs = './outputs/{}'.format(args.name)
    
    if local_rank == 0:
        tblogger = SummaryWriter(os.path.join(outputs, "tensorboard"))

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    custom_datasets = Dataset(img_size=args.img_size,
                        batch_size=args.batch_size,
                        workers=args.workers,
                        isDistributed=milti_gpus,
                        data_dir=args.data,
                        dataset_type='voc')
    train_loader, val_loader, train_sampler = custom_datasets.dataloader()

    best_acc1 = 0

    max_iter = len(train_loader)

    n_burnin = min(round(max_iter / 5 + 1), 1000)  # burn-in batches

    for epoch in range(args.start_epoch, args.epochs):
        if milti_gpus:
            # 使多顯卡訓練的訓練資料洗牌
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)
        freeze_backbone = True
        cutoff = 75
        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                # if int(name.split('.')[1]) < cutoff:  # if layer < 75
                #     p.requires_grad = False if epoch == 0 else True
                if name.split('.')[0] == 'backnone':
                    p.requires_grad = False if epoch == 0 else True
        
        # train for one epoch
        losses, xy_loss, wh_loss, obj_loss, cls_loss = train(train_loader,
                                                             model,
                                                             optimizer,
                                                             epoch,
                                                             args,
                                                             local_rank,
                                                             max_iter,
                                                             args.epochs,
                                                             n_burnin)
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
            if (epoch + 1) % 50 == 0:
                mAP = 0
                print('*'*20+"Validate"+'*'*20)
                with torch.no_grad():
                    APs = Evaluator(model,
                                    dataloader=val_loader,
                                    test_size=args.img_size,
                                    epoch=epoch+1, 
                                    conf_thres=0.01,
                                    nms_thresh=0.5).APs_voc()
                    for i in APs:
                        print("{} --> mAP : {}".format(i, APs[i]))
                        mAP += APs[i]
                    # num_classes = 20
                    mAP = mAP / 20
                    tblogger.add_scalar("val/mAP", mAP, epoch + 1)
                    print('mAP:%g'%(mAP))

        # evaluate on validation set
        # acc1 = validate(val_loader, model, criterion, args, device, local_rank)
        # if local_rank == 0:
        #     tblogger.add_scalar("val/Acc1", acc1, epoch + 1)

        # remember best acc@1 and save checkpoint
        # is_best = acc1 > best_acc1
        # best_acc1 = max(acc1, best_acc1)

        scheduler_LR.step()

def save_checkpoint(state, is_best, outputs='', epoch=''):
    filename = os.path.join(outputs, 'epoch_{}.pth'.format(epoch))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(outputs, 'model_best.pth'))

def train(train_loader, model, optimizer, epoch, args, rank, max_iter, max_epoch, n_burnin):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    xy_loss = AverageMeter()
    wh_loss = AverageMeter()
    obj_loss = AverageMeter()
    cls_loss = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (imgs, target, paths) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # imgpath ='/home/lab602.demo/.pipeline/10678031/myYolo/outputs/voc/results/img2'
        
        # for index, img in enumerate(imgs):
        #     print(type(img))
        #     print(img.shape)
        #     cv2.imwrite(os.path.join(imgpath, '{}.jpg'.format(paths[index])), img)
        #     input('test')

        _, _, w, h = imgs.shape


        if args.gpu is not None:
            imgs = imgs.cuda()
        target = target.cuda()

        # SGD burn-in
        if epoch == 0 and i <= n_burnin:
            lr = args.lr * (i / n_burnin) ** 4
            for x in optimizer.param_groups:
                x['lr'] = lr

        # compute output
        output = model(imgs)
        # print('-------------train-----------------')
        # print(output.shape)
        # loss = criterion(output, target)
        loss, loss_components = compute_loss(output, target, model)
        # print(loss_components.shape)

        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), imgs.size(0))
        xy_loss.update(loss_components[0].item(), imgs.size(0))
        wh_loss.update(loss_components[1].item(), imgs.size(0))
        obj_loss.update(loss_components[2].item(), imgs.size(0))
        cls_loss.update(loss_components[3].item(), imgs.size(0))
        # top1.update(acc1[0], img.size(0))
        # top5.update(acc5[0], img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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