import sys
import torch
import torch.nn as nn
import math
import numpy as np


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


def wh_iou(box1, box2):
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou


hyp = {'giou': 1.666,  # giou loss gain
       'xy': 4.062,  # xy loss gain
       'wh': 0.1845,  # wh loss gain
       'cls': 42.6,  # cls loss gain
       'cls_pw': 3.34,  # cls BCELoss positive_weight
       'obj': 12.61,  # obj loss gain
       'obj_pw': 8.338,  # obj BCELoss positive_weight
       'iou_t': 0.2705,  # iou target-anchor training threshold
       'lr0': 0.001,  # initial learning rate
       'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.90,  # SGD momentum
       'weight_decay': 0.0005}  # optimizer weight decay


def compute_loss(pred, targets, model):  # predictions, targets, model
    ft = torch.cuda.FloatTensor if pred[0].is_cuda else torch.Tensor
    lxy, lwh, lcls, lobj = ft([0]), ft([0]), ft([0]), ft([0])
    txy, twh, tcls, indices = build_targets(pred, model, targets)#在13 26 52維度中找到大於iou閾值最適合的anchor box 作為targets
    #txy[維度(0:2),(x,y)] twh[維度(0:2),(w,h)] indices=[0,anchor索引，gi，gj]

    # Define Loss
    MSE = nn.MSELoss()
    # CE = nn.CrossEntropyLoss()
    # BCE = nn.BCEWithLogitsLoss()
    # BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([hyp['cls_pw']]))
    # BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([hyp['obj_pw']]))

    BCEcls = nn.BCEWithLogitsLoss()
    BCEobj = nn.BCEWithLogitsLoss()

    # Compute losses
    # h = model.hyp  # hyperparameters
    batch_size = pred[0].shape[0]  # batch size
    k = batch_size / 64  # loss gain

    for i, pi0 in enumerate(pred):  # layer i predictions, i
        # pi0 = torch.Size([16, 3, 13, 13, 25])
        b, a, gj, gi = indices[i]  # image, anchor, gridx, gridy
        # tobj = torch.Size([16, 3, 13, 13])
        tobj = torch.zeros_like(pi0[..., 0])  # conf

        # Compute losses
        if len(b):  # number of targets
            pi = pi0[b, a, gj, gi]  # predictions closest to anchors 找到p中與targets對應的數據lxy
            # pi = torch.Size([num_target, 25])
            tobj[b, a, gj, gi] = 1  # conf
            # pi[..., 2:4] = torch.sigmoid(pi[..., 2:4])  # wh power loss (uncomment)

            # torch.Size([num_target, 2])
            lxy += MSE(torch.sigmoid(pi[..., 0:2]), txy[i])  # xy loss
            lwh += MSE(pi[..., 2:4], twh[i])  # wh yolo loss

            # tclsm = torch.Size([num_target, 20])
            tclsm = torch.zeros_like(pi[..., 5:])
            tclsm[range(len(b)), tcls[i]] = 1.0
            lcls += BCEcls(pi[..., 5:], tclsm)  # class_conf loss
            # lcls += (k * hyp['cls']) * CE(pi[..., 5:], tcls[i])  # cls loss (CE)

        # pos_weight = ft([gp[i] / min(gp) * 4.])
        # BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        lobj += BCEobj(pi0[..., 4], tobj)  # obj_conf loss
    
    # lxy *= (k * hyp['xy'])
    # lwh *= (k * hyp['wh'])
    # lobj *= (k * hyp['cls'])
    # lcls *= (k * hyp['obj'])

    lxy *= 0.1
    lwh *= 0.2
    lobj *= 1.0
    lcls *= 0.5
    loss = lxy + lwh + lobj + lcls

    return loss, torch.cat((lxy, lwh, lobj, lcls, loss)).detach()



def build_targets(pred, model, targets):
    # pred = [0_batch_size, 1_self.num_anchors, 2_nG, 3_nG, 4_5+self.num_classes]
    # targets = [image, class, x(歸一後的中心), y, w（歸一後的寬）, h] [ 0.00000, 20.00000,  0.72913,  0.48770,  0.13595,  0.08381]
    # iou_thres = model.hyp['iou_t']  # hyperparameter
    iou_thres = 0.5
    # if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel):
    #     model = model.module

    # ANCHORS = [[(10,13), (16,30), (33,23)],  # Anchors for small obj
    #             [(30,61, ), (62,45), (59,119)],  # Anchors for medium obj
    #             [(116,90), (156,198), (373,326)]]# Anchors for big obj
    
    m = de_parallel(model)
    num_anchors = m.num_anchors
    # anchors = torch.Tensor(ANCHORS).to(pred[0].device)
    anchors = m.anchors
    strides = m.strides
    num_layers = m.num_layers
    num_classes = m.num_classes

    # print('12312313')
    # print(targets.shape)

    num_targets = len(targets)
    txy, twh, tcls, indices = [], [], [], []
    # for i in model.yolo_layers:
    for i in range(num_layers):
        anchor = anchors[i, :, :] / strides[i]
        feature_size = pred[i].shape[2]
        # iou of targets-anchors
        t, anchor_i = targets, []
        gwh = targets[:, 4:6] * feature_size  # nG layer.ng就是yolo層輸出維度13  26  52，gwh將原來的wh還原到13*13的圖上
        if num_targets:
            # 計算3 anchor對應的iou
            iou = [wh_iou(x, gwh) for x in anchor]
            iou, anchor_i = torch.max(torch.stack(iou, dim=0), dim=0)  # best iou and anchor找到每一層與label->wh，iou最大的anchor,a是anchor的索引

            # reject below threshold ious (OPTIONAL, increases P, lowers R)
            reject = True
            if reject:
                j = iou > iou_thres
                t, anchor_i, gwh = targets[j], anchor_i[j], gwh[j]

        # Indices targets = [image, class, x(歸一後的中心), y, w（歸一後的寬）, h]
        # torch.Size([num_targets, 2]) -> torch.Size([2, num_targets])
        b, c = t[:, :2].long().t()  # target image, class

        # torch.Size([num_targets, 2])
        gxy = t[:, 2:4] * feature_size # 表示真實框的x軸y軸座標

        # torch.Size([num_targets, 2]) -> torch.Size([2, num_targets])
        gi, gj = gxy.long().t()  # grid_i, grid_j 表示真實框對應特徵點的x軸y軸座標
        indices.append((b, anchor_i, gj, gi))
        # XY coordinates
        txy.append(gxy - gxy.floor())#在yolov3裡是Gx,Gy減去grid cell左上角坐標Cx,Cy
        # Width and height
        twh.append(torch.log(gwh / anchor[anchor_i]))  # wh yolo method
        # twh.append((gwh / layer.anchor_vec[a]) ** (1 / 3) / 2)  # wh power method
        # Class
        tcls.append(c)
        # print(c.shape)
        # print(c.shape[0])
        # print(c)
        if c.shape[0]:
            assert c.max() <= num_classes, 'Target classes exceed model classes'

    return txy, twh, tcls, indices

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