import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bn=None):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=not bn)
        # self.bn = bn
        if bn is not None:
            self.test = True
            self.bn = nn.BatchNorm2d(out_channels)
            self.leakyreLU = nn.LeakyReLU(0.1)
        else:
            self.test = False

    def forward(self, x):
        x = self.conv(x)
        if self.test:
            x = self.bn(x)
            x = self.leakyreLU(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        assert in_channels % 2 == 0  # ensure the in_channels is even
        half_in_channels = in_channels // 2
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=half_in_channels, kernel_size=1, stride=1, padding=0, bn='bn')
        self.conv2 = Conv2d(in_channels=half_in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bn='bn')

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out


class StackResidualBlock(nn.Module):
    def __init__(self, in_channels, num_block):
        super(StackResidualBlock, self).__init__()
        self.sequential = nn.Sequential()
        for i in range(num_block):
            self.sequential.add_module('stack_%d' % (i+1,), ResidualBlock(in_channels=in_channels))

    def forward(self, x):
        for residual_block in self.sequential:
            x = residual_block(x)
        return x


class Darknet53(nn.Module):
    def __init__(self, in_channels=3, pretrained=False):
        super(Darknet53, self).__init__()
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1, bn='bn')

        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bn='bn')
        self.stack_residual_block_1 = StackResidualBlock(in_channels=64, num_block=1)

        self.conv3 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bn='bn')
        self.stack_residual_block_2 = StackResidualBlock(in_channels=128, num_block=2)

        self.conv4 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bn='bn')
        self.stack_residual_block_3 = StackResidualBlock(in_channels=256, num_block=8)

        self.conv5 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bn='bn')
        self.stack_residual_block_4 = StackResidualBlock(in_channels=512, num_block=8)

        self.conv6 = Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1, bn='bn')
        self.stack_residual_block_5 = StackResidualBlock(in_channels=1024, num_block=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.stack_residual_block_1(x)
        x = self.conv3(x)
        x = self.stack_residual_block_2(x)
        x = self.conv4(x)
        dark3 = self.stack_residual_block_3(x)
        x = self.conv5(dark3)
        dark4 = self.stack_residual_block_4(x)
        x = self.conv6(dark4)
        dark5 = self.stack_residual_block_5(x)

        return dark3, dark4, dark5


class Darknet19(nn.Module):
    def __init__(self, in_channels=3, pretrained=False):
        super(Darknet19, self).__init__()
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1, bn='bn')
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bn='bn')
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bn='bn')
        self.conv4 = Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bn='bn')
        self.conv5 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bn='bn')
        
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bn='bn')
        self.conv7 = Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bn='bn')
        self.conv8 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bn='bn')
        
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv9 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bn='bn')
        self.conv10 = Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bn='bn')
        self.conv11 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bn='bn')
        self.conv12 = Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bn='bn')
        self.conv13 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bn='bn')
        
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv14 = Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bn='bn')
        self.conv15 = Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bn='bn')
        self.conv16 = Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bn='bn')
        self.conv17 = Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bn='bn')
        self.conv18 = Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bn='bn')

    def forward(self, x):
        x = self.conv1(x)

        x = self.maxpool1(x)
        x = self.conv2(x)

        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.maxpool3(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        x = self.maxpool4(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        x = self.maxpool5(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        out = self.conv18(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_weight(self):
        weight_file = 'weights/darknet19-deepBakSu-e1b3ec1e.pth'
        # 轉換權重文件中的keys.(change the weights dict `keys)
        assert len(torch.load(weight_file).keys()) == len(self.state_dict().keys())
        dic = {}
        for now_keys, values in zip(self.state_dict().keys(), torch.load(weight_file).values()):
            dic[now_keys]=values
        self.load_state_dict(dic)

if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Darknet19()
    model = model.to(device)

    summary(model, (3, 224, 224))
 
    params=model.state_dict()

    for k,v in params.items():
        print(k)