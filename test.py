from models.detector import YOLOv3
from models.eval.evaluator import Evaluator
from loguru import logger
import argparse
import os
import torch
import yaml


class Tester(object):
    def __init__(self,
                 weight_path=None,
                 gpu_id=0,
                 img_size=544,
                 ):
        self.img_size = img_size
        self.device = torch.device('cuda:{}'.format(gpu_id) if True else 'cpu')
        outputs = './outputs/voc'
        logger.add(f'{outputs}/train_log.txt', encoding='utf-8', enqueue=True)
        self.multi_scale_test = False
        self.flip_test = False

        with open('configs/yolov3.yaml', errors='ignore') as f:
            configs = yaml.safe_load(f)

        self.num_classes = configs['nc']
        self.model = YOLOv3(anchors=configs['anchors'],
                            strides=configs['strides'][0],
                            num_classes=self.num_classes,
                            pretrained=True).to(self.device)

        self.load_model_weights(weight_path)

    def load_model_weights(self, weight_path):
        logger.info('loading weight file from : {}'.format(weight_path))
        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.device)
        self.model.load_state_dict(chkpt['model'])
        logger.info('loading weight file is done')
        del chkpt

    def test(self):
        mAP = 0
        logger.info('*'*20+"Validate"+'*'*20)

        with torch.no_grad():
            APs = Evaluator(self.model).APs_voc(self.multi_scale_test, self.flip_test)
            for i in APs:
                logger.info(f'{i} --> mAP : {APs[i]}')
                mAP += APs[i]
            mAP = mAP / self.num_classes
            logger.info(f'mAP:{mAP}')


if __name__ == "__main__":
    # 'outputs/voc/epoch_20.pth'
    # 'outputs/last.pth'
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='outputs/voc/epoch_20.pth', help='weight file path')
    parser.add_argument('--visiual', type=str, default='outputs/test', help='test data path or None')
    parser.add_argument('--eval', action='store_true', default=True, help='eval the mAP or not')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    opt = parser.parse_args()

    Tester( weight_path=opt.weight_path,
            gpu_id=opt.gpu_id).test()
