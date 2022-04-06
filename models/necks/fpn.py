import sys,os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.darknet import Conv2d


class Upsample(nn.Module):
    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x1, x2):
        """
        x1 means previous output; x2 means current output
        """
        out = torch.cat((x2, x1), dim=1)
        return out


class YOLOFPN(nn.Module):
    """
    FPN for yolov3, and is different from original FPN or retinanet' FPN.
    """
    def __init__(self, in_channels=[1024, 512, 256], out_channels=None):
        super(YOLOFPN, self).__init__()
        # large
        # FPN0 = 13*13*75
        self.large_conv1 = Conv2d(in_channels=in_channels[0], out_channels=512, kernel_size=1, stride=1, padding=0)
        self.large_conv2 = Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.large_conv3 = Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.large_conv4 = Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.large_conv5 = Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.large_conv6 = Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.large_conv7 = nn.Conv2d(in_channels=1024, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        # medium
        # FPN1 = 26*26*75
        self.medium_conv1 = Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.upsample1 = Upsample(scale_factor=2)
        self.route1 = Route()

        self.medium_conv2 = Conv2d(in_channels=int(in_channels[1]*1.5), out_channels=256, kernel_size=1, stride=1, padding=0)
        self.medium_conv3 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.medium_conv4 = Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.medium_conv5 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.medium_conv6 = Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        
        self.medium_conv7 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.medium_conv8 = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        # small
        # FPN2 = 52*52*75
        self.small_conv1 = Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.upsample2 = Upsample(scale_factor=2)
        self.route2 = Route()

        self.small_conv2 = Conv2d(in_channels=int(in_channels[2]*1.5), out_channels=128, kernel_size=1, stride=1, padding=0)
        self.small_conv3 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.small_conv4 = Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.small_conv5 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.small_conv6 =Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.small_conv7 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.small_conv8 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x0, x1, x2):  # large, medium, small
        # large
        x0 = self.large_conv1(x0)
        x0 = self.large_conv2(x0)
        x0 = self.large_conv3(x0)
        x0 = self.large_conv4(x0)
        x0 = self.large_conv5(x0)

        fpn0 = self.large_conv6(x0)
        fpn0 = self.large_conv7(fpn0)


        # medium
        x0 = self.medium_conv1(x0)
        x0 = self.upsample1(x0)
        x1 = self.route1(x1, x0)
        x1 = self.medium_conv2(x1)
        x1 = self.medium_conv3(x1)
        x1 = self.medium_conv4(x1)
        x1 = self.medium_conv5(x1)
        x1 = self.medium_conv6(x1)

        fpn1 = self.medium_conv7(x1)
        fpn1 = self.medium_conv8(fpn1)


        # small
        x1 = self.small_conv1(x1)
        x1 = self.upsample2(x1)
        x2 = self.route2(x2, x1)
        x2 = self.small_conv2(x2)
        x2 = self.small_conv3(x2)
        x2 = self.small_conv4(x2)
        x2 = self.small_conv5(x2)
        x2 = self.small_conv6(x2)

        fpn2 = self.small_conv7(x2)
        fpn2 = self.small_conv8(fpn2)

        return fpn2, fpn1, fpn0  # small, medium, large