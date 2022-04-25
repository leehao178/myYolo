import sys,os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from models.backbones.darknet import Darknet53, Conv2d
from models.necks.fpn import YOLOFPN
from models.heads.yolo_head import YOLOHead
import numpy as np


class YOLOv3(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, anchors, strides, num_classes=20, num_anchors=3, num_layers=3, pretrained=None):
        super(YOLOv3, self).__init__()
        self.anchors = anchors
        self.strides = strides
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        out_channels = num_anchors * (num_classes + 5)

        self.backnone = Darknet53()
        self.fpn = YOLOFPN(in_channels=[1024, 512, 256], out_channels=out_channels)

        # small
        self.head_s = YOLOHead(num_classes=num_classes, anchors=anchors[0, :, :])
        # medium
        self.head_m = YOLOHead(num_classes=num_classes, anchors=anchors[1, :, :])
        # large
        self.head_l = YOLOHead(num_classes=num_classes, anchors=anchors[2, :, :])

        if pretrained is not None:
            self.load_darknet_weights(weight_file=pretrained)
        else:
            self.init_weights()
            

    def forward(self, x):
        out = []

        img_size = max(x.shape[-2:])

        dark3, dark4, dark5 = self.backnone(x)
        '''
        dark3 = torch.Size([16, 256, 52, 52])
        dark4 = torch.Size([16, 512, 26, 26])
        dark5 = torch.Size([16, 1024, 13, 13])
        '''

        x_small, x_medium, x_large = self.fpn(dark5, dark4, dark3)
        '''
        x_small = torch.Size([16, 75, 52, 52])
        x_medium = torch.Size([16, 75, 26, 26])
        x_large = torch.Size([16, 75, 13, 13])
        '''

        out.append(self.head_s(x_small, img_size))
        out.append(self.head_m(x_medium, img_size))
        out.append(self.head_l(x_large, img_size))
        '''
        head_s = torch.Size([16, 3, 52, 52, 25])
        head_m = torch.Size([16, 3, 26, 26, 25])
        head_l = torch.Size([16, 3, 13, 13, 25])
        '''

        if self.training:
            return out
        else:
            io, p = list(zip(*out))  # inference output, training output
            '''
            io[0] = torch.Size([16, 8112, 25])
            io[1] = torch.Size([16, 2028, 25])
            io[2] = torch.Size([16, 507, 25])
            p[0] = torch.Size([16, 3, 52, 52, 25])
            p[1] = torch.Size([16, 3, 26, 26, 25])
            p[2] = torch.Size([16, 3, 13, 13, 25])
            torch.cat(io, 1) = torch.Size([16, 10647, 25])
            '''
            return torch.cat(io, 1), p


    def init_weights(self):
        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

                print("initing {}".format(m))



    def load_darknet_weights(self, weight_file='/home/lab602.demo/.pipeline/10678031/myYolo/outputs/yolov3.weights', cutoff=52):
        "https://github.com/ultralytics/yolov3/blob/master/models.py"
        print("load darknet weights : ", weight_file)

        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0

        for m in self.modules():
            if isinstance(m, Conv2d):
                # only initing backbone conv's weights
                if count == cutoff:
                    break
                count += 1
                conv_layer = m.conv
                # print(m.bn)
                if hasattr(m, 'bn'):
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = m.bn
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                    print("loading bn_layer weight {}".format(bn_layer))
                else:
                    # Load conv. bias
                    # print(conv_layer.bias)
                    # torch.zeros(m.weight.size()[2:]).numel()
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                    print("loading conv_layer weight {}".format(conv_layer))
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

                


if __name__ == "__main__":
    # from torchsummary import summary

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Dtector(80, 3, False)
    # model = model.to(device)

    # summary(model, (3, 224, 224))
    # test_data = torch.rand(1, 3, 352, 352)
    # torch.onnx.export(model,                    #model being run
    #                  test_data,                 # model input (or a tuple for multiple inputs)
    #                  "test.onnx",               # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=11,          # the ONNX version to export the model to
    #                  do_constant_folding=True)  # whether to execute constant folding for optimization
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ANCHORS = [[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj
                [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
                [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]]# Anchors for big obj
    STRIDES = [8, 16, 32]
    ANCHORS_PER_SCLAE = 3
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv3(anchors=torch.FloatTensor(ANCHORS),
                   strides=torch.FloatTensor(STRIDES))
    # model = model.to(device)
    # print(model)

    in_img = torch.randn(12, 3, 416, 416)
    p = model(in_img)

    for i in range(3):
        print(p[i].shape)
        # print(p_d[i].shape)