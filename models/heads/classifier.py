import torch
import torch.nn as nn
from models.backbones.darknet import Darknet53


class Classifier(nn.Module):
    def __init__(self, num_classes, in_channels, pretrained=False):
        super(Classifier, self).__init__()

        self.backbone = Darknet53(in_channels=in_channels, pretrained=pretrained)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)


    def forward(self, x):
        _, _, dark5 = self.backbone(x)
        out = self.global_avg_pool(dark5)
        out = out.view(-1, 1024)
        out = self.fc(out)

        return out


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classifier(80, 3, False)
    model = model.to(device)

    summary(model, (3, 224, 224))
    # test_data = torch.rand(1, 3, 352, 352)
    # torch.onnx.export(model,                    #model being run
    #                  test_data,                 # model input (or a tuple for multiple inputs)
    #                  "test.onnx",               # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=11,          # the ONNX version to export the model to
    #                  do_constant_folding=True)  # whether to execute constant folding for optimization