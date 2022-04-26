import torch.nn as nn
import torch


class YOLOHead(nn.Module):
    def __init__(self, num_classes, anchors):
        super(YOLOHead, self).__init__()

        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes

        # stride = 一個特徵點對應原來影像上的stride個像素點
        # self.stride = stride
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints

    def forward(self, p, img_size):
        batch_size, ny, nx = p.shape[0], p.shape[-2], p.shape[-1]
        if (self.nx, self.ny) != (nx, ny):
            self.create_grids(img_size, (nx, ny), p.device)
        # batch_size, nG = p.shape[0], p.shape[-1]

        # [0_batch_size, 1_nG, 2_nG, 3_self.num_anchors, 4_5+self.num_classes]

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(batch_size, self.num_anchors, 5 + self.num_classes, self.ny, self.nx).permute(0, 1, 3, 4, 2)

        # [0_batch_size, 1_self.num_anchors, 2_nG, 3_nG, 4_5+self.num_classes]
        if self.training:
            return p
        else:
            io = p.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., 4:] = torch.sigmoid(io[..., 4:])  # p_conf, p_cls
            # io[..., 5:] = F.softmax(io[..., 5:], dim=4)  # p_cls
            io[..., :4] *= self.stride
            if self.num_classes == 1:
                io[..., 5] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return io.reshape(batch_size, -1, 5 + self.num_classes), p

    def create_grids(self, img_size=416, ng=(13, 13), device='cpu'):
        nx, ny = ng  # x and y grid size
        self.img_size = img_size
        self.stride = img_size / max(ng)

        # build xy offsets
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        self.grid_xy = torch.stack((xv, yv), 2).to(device).float().view((1, 1, ny, nx, 2))

        # build wh gains
        self.anchor_vec = self.anchors.to(device) / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.num_anchors, 1, 1, 2).to(device)
        self.ng = torch.Tensor(ng).to(device)
        self.nx = nx
        self.ny = ny