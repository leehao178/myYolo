import os
import argparse
import time
import shutil
import torch
from torch import nn
from models.heads.classifier import Classifier
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import datetime
from models.data.datasets.dataset import Dataset


# '/home/lab602.demo/.pipeline/datasets/mnist'
# '/home/lab602.demo/.pipeline/datasets/ImageNet/ILSVRC2012'
parser = argparse.ArgumentParser(description='Netowks Classification Training')
parser.add_argument('-n', '--name', default='imagenet', type=str, help='name')
parser.add_argument('--data', default='/home/lab602.demo/.pipeline/datasets/ImageNet/ILSVRC2012',
                    type=str,
                    metavar='DIR',
                    help='path to dataset')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
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


def main():
    args = parser.parse_args()
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
            torch.cuda.set_device(args.gpu)
            print("Use GPU: {} for training".format(args.gpu))

    model = Classifier(num_classes=1000, in_channels=3, pretrained=False)

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
    criterion = nn.CrossEntropyLoss().cuda()
    # if milti_gpus:
    #     args.lr = args.lr * torch.cuda.device_count()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    scheduler_LR = StepLR(optimizer, step_size=1, gamma=0.1)
    
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

    custom_datasets = Dataset(img_size=224,
                        batch_size=args.batch_size,
                        workers=args.workers,
                        isDistributed=True,
                        data_dir=args.data,
                        MNIST=False)
    train_loader, val_loader, train_sampler = custom_datasets.dataloader()

    best_acc1 = 0

    max_iter = len(train_loader)

    for epoch in range(args.start_epoch, args.epochs):
        if milti_gpus:
            # 使多顯卡訓練的訓練資料洗牌
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        acc1, acc5, losses = train(train_loader, model, criterion, optimizer, epoch, args, local_rank, max_iter, args.epochs)
        if local_rank == 0:
            tblogger.add_scalar("train/loss", losses, epoch + 1)
            tblogger.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], epoch + 1)
            tblogger.add_scalar("train/Acc1", acc1, epoch + 1)
            tblogger.add_scalar("train/Acc5", acc5, epoch + 1)
        
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, device, local_rank)
        if local_rank == 0:
            tblogger.add_scalar("val/Acc1", acc1, epoch + 1)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if local_rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, outputs=outputs, epoch=epoch+1)
        scheduler_LR.step()

def save_checkpoint(state, is_best, outputs='', epoch=''):
    filename = os.path.join(outputs, 'epoch_{}.pth'.format(epoch))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(outputs, 'model_best.pth'))

def train(train_loader, model, criterion, optimizer, epoch, args, rank, max_iter, max_epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

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
                print('Epoch: [{0}][{1}/{2}]\t'
                    'ETA: {eta}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Learing Rate {LR}\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch+1, i, len(train_loader), eta=datetime.timedelta(seconds=int(eta_seconds)), batch_time=batch_time,
                    data_time=data_time, loss=losses, LR=optimizer.param_groups[0]['lr'], top1=top1, top5=top5))
    return top1.avg, top5.avg, losses.avg

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

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def validate(val_loader, model, criterion, args, device, rank):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if rank == 0:
                if i % args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))
        if rank == 0:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return top1.avg

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