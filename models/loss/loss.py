import sys
import torch
import torch.nn as nn
import math
import numpy as np
from models.utils.bbox import iou_xywh_torch

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma)

        return loss

def to_cpu(tensor):
    return tensor.detach().cpu()

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


class YOLOv3Loss(object):
    def __init__(self, iou_threshold_loss=0.5):
        self.iou_threshold_loss = iou_threshold_loss

    def __call__(self, p, p_d, bboxes_xywh, label, model):
        """
        分層計算loss值。
        :param p: 預測偏移值。 shape為[p0, p1, p2]共三個檢測層，其中以p0為例，其shape為(bs,  grid, grid, anchors, tx+ty+tw+th+conf+cls_20)
        :param p_d: 解碼後的預測值。 shape為[pd0, pd1, pd2]，其中以pd0為例，其shape為(bs,  grid, grid, anchors, x+y+w+h+conf+cls_20)
        :param label_sbbox: small 檢測層的分配標籤, shape為[bs,  grid, grid, anchors, x+y+w+h+conf+cls_20]
        :param label_mbbox: medium檢測層的分配標籤, shape為[bs,  grid, grid, anchors, x+y+w+h+conf+cls_20]
        :param label_lbbox: large檢測層的分配標籤, shape為[bs,  grid, grid, anchors, x+y+w+h+conf+cls_20]
        :param sbboxes: small檢測層的bboxes, shape為[bs, 150, x+y+w+h]
        :param mbboxes: medium檢測層的bboxes, shape為[bs, 150, x+y+w+h]
        :param lbboxes: large檢測層的bboxes, shape為[bs, 150, x+y+w+h]
        :return: loss為總損失值，loss_l[m, s]為每層的損失值
        """
        m = de_parallel(model)
        num_anchors = m.num_anchors
        anchors = m.anchors
        strides = m.strides
        num_layers = m.num_layers
        num_classes = m.num_classes

        loss_s, loss_s_xywh, loss_s_conf, loss_s_cls = self.cal_loss(p[0], p_d[0], label[0], bboxes_xywh[0], anchors[0, :, :] / strides[0], strides[0])
        loss_m, loss_m_xywh, loss_m_conf, loss_m_cls = self.cal_loss(p[1], p_d[1], label[1], bboxes_xywh[1], anchors[1, :, :] / strides[1], strides[1])
        loss_l, loss_l_xywh, loss_l_conf, loss_l_cls = self.cal_loss(p[2], p_d[2], label[2], bboxes_xywh[2], anchors[2, :, :] / strides[2], strides[2])
        loss = (loss_l + loss_m + loss_s) / 3
        loss_xywh = (loss_s_xywh + loss_m_xywh + loss_l_xywh) / 3
        loss_conf = (loss_s_conf + loss_m_conf + loss_l_conf) / 3
        loss_cls = (loss_s_cls + loss_m_cls + loss_l_cls) / 3

        return loss, loss_xywh, loss_conf, loss_cls


    def cal_loss(self, p, p_d, label, bboxes, anchors, stride):
        """
        計算每一層的損失。損失由三部分組成，(1)boxes的回歸損失。計算預測的偏移量和標籤的偏移量之間的損失。其中
        首先需要將標籤的坐標轉換成該層的相對於每個網格的偏移量以及長寬相對於每個anchor的比例係數。
        注意：損失的係數為2-w*h/(img_size**2),用於在不同尺度下對損失值大小影響不同的平衡。
        (2)置信度損失。包括前景和背景的置信度損失值。其中背景損失值需要注意的是在某特徵點上標籤為背景並且該點預測的anchor
        與該image所有bboxes的最大iou小於閾值時才計算該特徵點的背景損失。
        (3)類別損失。類別損失為BCE，即每個類的二分類值。
        :param p: 沒有進行解碼的預測值，表示形式為(bs,  grid, grid, anchors, tx+ty+tw+th+conf+classes)
        :param p_d: p解碼以後的結果。 xywh均為相對於原圖的尺度和位置，conf和cls均進行sigmoid。表示形式為
        [bs, grid, grid, anchors, x+y+w+h+conf+cls]
        :param label: lable的表示形式為(bs,  grid, grid, anchors, x+y+w+h+conf+cls) 其中xywh均為相對於原圖的尺度和位置。
        :param bboxes: 該batch內分配給該層的所有bboxes，shape為(bs, 150, 4).
        :param anchors: 該檢測層的ahchor尺度大小。格式為torch.tensor
        :param stride: 該層feature map相對於原圖的尺度縮放量
        :return: 該檢測層的所有batch平均損失。 loss=loss_xywh + loss_conf + loss_cls。
        """
        BCE = nn.BCEWithLogitsLoss(reduction="none")
        MSE = nn.MSELoss(reduction="none")

        batch_size, grid = p.shape[:2]
        img_size = stride * grid
        device = p.device

        p_dxdy = p[..., 0:2]
        p_dwdh = p[..., 2:4]
        p_conf = p[..., 4:5]
        p_cls = p[..., 5:]

        p_d_xywh = p_d[..., :4]  # 用於計算iou


        label_xywh = label[..., :4]
        label_obj_mask = label[..., 4:5]
        label_cls = label[..., 5:]

        # loss xywh
        ## label的坐標轉換為tx,ty,tw,th
        y = torch.arange(0, grid).unsqueeze(1).repeat(1, grid)
        x = torch.arange(0, grid).unsqueeze(0).repeat(grid, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)


        label_txty = (1.0 * label_xywh[..., :2] / stride) - grid_xy
        label_twth = torch.log((1.0 * label_xywh[..., 2:] / stride) / anchors.to(device))
        label_twth = torch.where(torch.isinf(label_twth), torch.zeros_like(label_twth), label_twth)

        # bbox的尺度縮放權值
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (img_size ** 2)

        loss_xy = label_obj_mask * bbox_loss_scale * BCE(input=p_dxdy, target=label_txty)
        loss_wh = 0.5 * label_obj_mask * bbox_loss_scale * MSE(input=p_dwdh, target=label_twth)


        # loss confidence
        iou = iou_xywh_torch(p_d_xywh.unsqueeze(4), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        iou_max = iou.max(-1, keepdim=True)[0]
        label_noobj_mask = (1.0 - label_obj_mask) * (iou_max < self.iou_threshold_loss).float()

        loss_conf = label_obj_mask * BCE(input=p_conf, target=label_obj_mask) + \
                    label_noobj_mask * BCE(input=p_conf, target=label_obj_mask)

        # loss classes
        loss_cls = label_obj_mask * BCE(input=p_cls, target=label_cls)

        # loss = torch.cat([loss_xy, loss_wh, loss_conf, loss_cls], dim=-1)
        # loss = loss.sum([1,2,3,4], keepdim=True).mean()  # batch的平均損失

        # loss_xywh = torch.cat([loss_xy, loss_wh], dim=-1)

        loss_xywh = (torch.sum(loss_xy) + torch.sum(loss_wh)) / batch_size
        loss_conf = (torch.sum(loss_conf)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size

        loss = loss_xywh + loss_conf + loss_cls

        return loss, loss_xywh, loss_conf, loss_cls


if __name__ == "__main__":
    import sys,os
    sys.path.append(os.getcwd())
    from models.detector import YOLOv3
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = YOLOv3()
    # model = model.to(device)
    p, p_d = model(torch.rand(3, 3, 416, 416))
    label_sbbox = torch.rand(3,  52, 52, 3,26)
    label_mbbox = torch.rand(3,  26, 26, 3, 26)
    label_lbbox = torch.rand(3, 13, 13, 3,26)
    sbboxes = torch.rand(3, 150, 4)
    mbboxes = torch.rand(3, 150, 4)
    lbboxes = torch.rand(3, 150, 4)

    ANCHORS = [[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj
                [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
                [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]]# Anchors for big obj
    STRIDES = [8, 16, 32]
    ANCHORS_PER_SCLAE = 3

    loss, loss_xywh, loss_conf, loss_cls = YoloV3Loss(ANCHORS, STRIDES)(p, p_d, label_sbbox,
                                    label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)
    print(loss)