import sys,os
sys.path.append(os.getcwd())
import torch
import numpy as np
from models.backbones.darknet import Darknet53, Conv2d
from models.detector import YOLOv3


def main():
    ANCHORS = [[(10,13), (16,30), (33,23)],  # Anchors for small obj
                [(30,61, ), (62,45), (59,119)],  # Anchors for medium obj
                [(116,90), (156,198), (373,326)]]# Anchors for big obj
    STRIDES = [8, 16, 32]
    path = '/home/lab602.demo/.pipeline/10678031/myYolo/outputs/yolov3.weights'
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device('cuda:{}'.format(gpu) if True else 'cpu')
    print("Use GPU: {} for training".format(gpu))
    model = YOLOv3(anchors=torch.FloatTensor(ANCHORS).to(device),
                   strides=torch.FloatTensor(STRIDES).to(device), num_classes=80)
    
    model.load_darknet_weights(weight_file=path, cutoff=75)
    torch.save(model.state_dict(), 'yolov3_good.pth')




if __name__ == '__main__':
    main()