import argparse
import torch
from models.data.datasets.dataset import Dataset
from loguru import logger
from models.detector import YOLOv3
from models.eval.evaluator import Evaluator
import yaml


parser = argparse.ArgumentParser(description='Netowks Object Detection Test')
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
parser.add_argument('--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--gpu', default=2, type=int,
                    help='GPU id to use.')
parser.add_argument("--local_rank", type=int, default=0,
                    help='node rank for distributed training')
parser.add_argument("-c", "--cpkt", type=str, default='outputs/voc/epoch_50.pth',
                    help='pth')


def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if True else 'cpu')
    logger.add('outputs/voc/test_log.txt', encoding='utf-8', enqueue=True)
    logger.info(f'Use GPU: {args.gpu} for training')

    with open(args.config_file, errors='ignore') as f:
        configs = yaml.safe_load(f)

    model = YOLOv3(anchors=torch.FloatTensor(configs['anchors']).to(device),
                   strides=torch.FloatTensor(configs['strides'][0]).to(device))
    
    model.cuda(args.gpu)
    model.load_state_dict(torch.load(args.cpkt, map_location=device)['model'])
    custom_datasets = Dataset(img_size=configs['test_size'],
                        batch_size=args.batch_size,
                        workers=args.workers,
                        isDistributed=False,
                        data_dir=args.data,
                        dataset_type='voc')
    _, val_loader, _ = custom_datasets.dataloader()

    mAP = 0
    with torch.no_grad():
        APs = Evaluator(model,
                        dataloader=val_loader,
                        configs=configs).APs_voc()
        for i in APs:
            logger.info(f'{i} --> mAP : {APs[i]}')
            mAP += APs[i]
        mAP = mAP / configs['nc']
        logger.info(f'mAP:{mAP}')


if __name__ == '__main__':
    main()