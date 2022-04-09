# import config.yolov3_config_voc as cfg
import os
import shutil
from models.eval.voc_eval import voc_eval
# from utils.datasets import *
# from utils.gpu import *
import cv2
import numpy as np
# from utils.data_augment import *
import torch
# from utils.tools import *
from tqdm import tqdm
# from utils.visualize import *
from models.utils.visualization import _COLORS

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

            # Plot
            # plt.plot(recall_curve, precision_curve)

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    iou = inter_area / union_area  # iou
    if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
        c_x1, c_x2 = torch.min(b1_x1, b2_x1), torch.max(b1_x2, b2_x2)
        c_y1, c_y2 = torch.min(b1_y1, b2_y1), torch.max(b1_y2, b2_y2)
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1)  # convex area
        return iou - (c_area - union_area) / c_area  # GIoU

    return iou

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_conf, class)
    """

    min_wh = 2  # (pixels) minimum box width and height

    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # Experiment: Prior class size rejection
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # a = w * h  # area
        # ar = w / (h + 1e-16)  # aspect ratio
        # n = len(w)
        # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] =
        #   multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])
        # print('=====================')
        # print(pred.shape)
        # Multiply conf by class conf to get combined confidence
        class_conf, class_pred = pred[:, 5:].max(1)
        pred[:, 4] *= class_conf
        # conf_thres = 0.00001
        # min_wh = 0
        
        # Select only suitable predictions
        i = (pred[:, 4] > conf_thres) & (pred[:, 2:4] > min_wh).all(1) & torch.isfinite(pred).all(1)
        pred = pred[i]
        
        # If none are remaining => process next image
        if len(pred) == 0:
            continue

        # Select predicted classes
        class_conf = class_conf[i]
        class_pred = class_pred[i].unsqueeze(1).float()

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])
        # pred[:, 4] *= class_conf  # improves mAP from 0.549 to 0.551

        # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
        pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)

        # Get detections sorted by decreasing confidence scores
        pred = pred[(-pred[:, 4]).argsort()]

        
        # print(pred)

        det_max = []
        nms_style = 'MERGE'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        for c in pred[:, -1].unique():
            dc = pred[pred[:, -1] == c]  # select class c
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                continue
            elif n > 100:
                dc = dc[:100]  # limit to first 100 boxes: https://github.com/ultralytics/yolov3/issues/117

            # Non-maximum suppression
            if nms_style == 'OR':  # default
                # METHOD1
                # ind = list(range(len(dc)))
                # while len(ind):
                # j = ind[0]
                # det_max.append(dc[j:j + 1])  # save highest conf detection
                # reject = (bbox_iou(dc[j], dc[ind]) > nms_thres).nonzero()
                # [ind.pop(i) for i in reversed(reject)]

                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

            elif nms_style == 'SOFT':  # soft-NMS https://arxiv.org/abs/1704.04503
                sigma = 0.5  # soft-nms sigma parameter
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    det_max.append(dc[:1])
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:]
                    dc[:, 4] *= torch.exp(-iou ** 2 / sigma)  # decay confidences
                    # dc = dc[dc[:, 4] > nms_thres]  # new line per https://github.com/ultralytics/yolov3/issues/362

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort

    return output


class Resize(object):
    """
    Resize the image to target size and transforms it into a color channel(BGR->RGB),
    as well as pixel value normalization([0,1])
    """
    def __init__(self, target_shape, correct_box=True):
        self.h_target, self.w_target = target_shape
        self.correct_box = correct_box

    def __call__(self, img, bboxes):
        h_org , w_org , _= img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        resize_ratio = min(1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org)
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)
        image_resized = cv2.resize(img, (resize_w, resize_h))

        image_paded = np.full((self.h_target, self.w_target, 3), 128.0)
        dw = int((self.w_target - resize_w) / 2)
        dh = int((self.h_target - resize_h) / 2)
        image_paded[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized
        image = image_paded / 255.0  # normalize to [0, 1]

        if self.correct_box:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
            return image, bboxes
        return image

def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def iou_xyxy_numpy(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要滿足廣播機制
    :param boxes2: 且需要保證最後一維為坐標維，以及坐標的存儲結構為(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape為boxes1和boxes2廣播後的shape[:-1]
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 計算出boxes1和boxes2相交部分的左上角坐標、右下角坐標
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 計算出boxes1和boxes2相交部分的寬、高
    # 因為兩個boxes沒有交集時，(right_down - left_up) < 0，所以maximum可以保證當兩個boxes沒有交集時，它們之間的iou為0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU

def nms(bboxes, score_threshold, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes:
    假設有N個bbox的score大於score_threshold，那麼bboxes的shape為(N, 6)，存儲格式為(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相對於輸入原圖的，score = conf * prob，class是bbox所屬類別的索引號
    :return: best_bboxes
    假設NMS後剩下N個bbox，那麼best_bboxes的shape為(N, 6)，存儲格式為(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相對於輸入原圖的，score = conf * prob，class是bbox所屬類別的索引號
    """
    classes_in_img = list(set(bboxes[:, 5].astype(np.int32)))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5].astype(np.int32) == cls)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = iou_xyxy_numpy(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            assert method in ['nms', 'soft-nms']
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > score_threshold
            cls_bboxes = cls_bboxes[score_mask]
    return np.array(best_bboxes)


class Evaluator(object):
    def __init__(self, model, dataloader, visiual=True):
        self.classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
        self.pred_result_path = os.path.join('/home/lab602.demo/.pipeline/10678031/myYolo/outputs/voc', 'results')
        self.val_data_path_07 = os.path.join('/home/lab602.demo/.pipeline/datasets/VOCdevkit', 'VOC2007')
        self.val_data_path_12 = os.path.join('/home/lab602.demo/.pipeline/datasets/VOCdevkit', 'VOC2012')
        self.conf_thresh = 0.01
        self.nms_thresh = 0.5
        self.val_shape =  416

        self.__visiual = visiual
        self.__visual_imgs = 0
        self.dataloader = dataloader

        self.model = model
        self.device = next(model.parameters()).device

    def APs_voc(self, multi_test=False, flip_test=False, save_json=False):
        self.model.eval()
        nc = len(self.classes)  # number of classes
        conf_thres = 0.001
        nms_thres = 0.5
        iou_thres = 0.5
        seen = 0
        # img_inds_file = os.path.join(self.val_data_path,  'ImageSets', 'Main', 'test.txt')
        # with open(img_inds_file, 'r') as f:
        #     lines = f.readlines()
        #     img_inds = [line.strip() for line in lines]

        if os.path.exists(self.pred_result_path):
            shutil.rmtree(self.pred_result_path)
        os.mkdir(self.pred_result_path)
        os.mkdir(os.path.join(self.pred_result_path, 'img'))
        print(('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1'))
        loss, p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0., 0.
        jdict, stats, ap, ap_class = [], [], [], []

        for batch_i, (imgs, targets, img_ids) in enumerate(tqdm(self.dataloader, desc='Computing mAP')):
            targets = targets.to(self.device)
            imgs = imgs.to(self.device)
            _, _, height, width = imgs.shape  # batch size, channels, height, width
            # print(len(imgs))
            # print(len(paths))
            # img_ind = paths.split('/')[-1].split('.')[0]

            # Plot images with bounding boxes
            # if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
            #     plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.jpg')

            # Run model
            inf_out, _ = self.model(imgs)  # inference and training outputs
            # print('--------------------')
            # print(inf_out.shape)
            # Compute loss
            # if hasattr(self.model, 'hyp'):  # if model has loss hyperparameters
            #     loss += compute_loss(train_out, targets, model)[0].item()

            # Run NMS
            output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)
            # print(output.shape)
            # print(output)
            print(img_ids)
            # Statistics per image
            for si, pred in enumerate(output):
                img_path = os.path.join(self.val_data_path_07, 'JPEGImages', f'{img_ids[si]}.jpg')
                img = cv2.imread(img_path)
                

                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if pred is None:
                    if nl:
                        stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                    continue
                
                for i in pred:
                   

                    # print(len(pred))
                    # print(len(i))
                    img_ind = 0
                    xmin, ymin, xmax, ymax, conf, score2, cls = i


                    color = (_COLORS[int(cls.cpu())] * 255).astype(np.uint8).tolist()
                    text = '{}:{:.1f}%'.format(self.classes[int(cls)], conf * 100)
                    txt_color = (0, 0, 0) if np.mean(_COLORS[int(cls.cpu())]) > 0.5 else (255, 255, 255)
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    txt_bk_color = (_COLORS[int(cls.cpu())] * 255 * 0.7).astype(np.uint8).tolist()
                    cv2.rectangle(
                        img,
                        (int(xmin), int(ymin) + 1),
                        (int(xmin) + txt_size[0] + 1, int(ymin) + int(1.5*txt_size[1])),
                        txt_bk_color,
                        -1
                    )
                    cv2.putText(img, text, (int(xmin), int(ymin) + txt_size[1]), font, 0.4, txt_color, thickness=1)
                    # print(conf)
                    # print(score2)
                    # class_ind = int(bbox[5])
                    class_name = self.classes[int(cls)]
                    s = ' '.join([str(img_ids[si]), str(float(conf)), str(float(xmin)), str(float(ymin)), str(float(xmax)), str(float(ymax))]) + '\n'

                    with open(os.path.join(self.pred_result_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
                        f.write(s)
                cv2.imwrite('./outputs/voc/results/img/{}.jpg'.format(img_ids[si]), img)
                # Append to text file
                # with open('test.txt', 'a') as file:
                #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

                # Append to pycocotools JSON dictionary
                # if save_json:
                #     # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                #     image_id = int(Path(paths[si]).stem.split('_')[-1])
                #     box = pred[:, :4].clone()  # xyxy
                #     scale_coords(imgs[si].shape[1:], box, shapes[si])  # to original shape
                #     box = xyxy2xywh(box)  # xywh
                #     box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                #     for di, d in enumerate(pred):
                #         jdict.append({
                #             'image_id': image_id,
                #             'category_id': coco91class[int(d[6])],
                #             'bbox': [float3(x) for x in box[di]],
                #             'score': float(d[4])
                #         })

                # Assign all predictions as incorrect
                # correct = [0] * len(pred)
                # if nl:
                #     detected = []
                #     tcls_tensor = labels[:, 0]

                #     # target boxes
                #     tbox = xywh2xyxy(labels[:, 1:5])
                #     tbox[:, [0, 2]] *= width
                #     tbox[:, [1, 3]] *= height

                #     # Search for correct predictions
                #     for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                #         # Break if all targets already located in image
                #         if len(detected) == nl:
                #             break

                #         # Continue if predicted class not among image classes
                #         if pcls.item() not in tcls:
                #             continue

                #         # Best iou, index between pred and targets
                #         m = (pcls == tcls_tensor).nonzero().view(-1)
                #         iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                #         # If iou > threshold and class is correct mark as correct
                #         if iou > iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                #             correct[i] = 1
                #             detected.append(m[bi])

                # # Append statistics (correct, conf, pcls, tcls)
                # stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

        # # Compute statistics
        # stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
        # nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        # if len(stats):
        #     p, r, ap, f1, ap_class = ap_per_class(*stats)
        #     mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()

        # # Print results
        # pf = '%20s' + '%10.3g' * 6  # print format
        # print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

        # # Print results per class
        # if nc > 1 and len(stats):
        #     for i, c in enumerate(ap_class):
        #         print(pf % (self.classes[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))


        # for img_ind in tqdm(img_inds):
        #     img_path = os.path.join(self.val_data_path, 'JPEGImages', img_ind+'.jpg')
        #     img = cv2.imread(img_path)
        #     bboxes_prd = self.get_bbox(img, multi_test, flip_test)

        #     if bboxes_prd.shape[0]!=0 and self.__visiual and self.__visual_imgs < 100:
        #         boxes = bboxes_prd[..., :4]
        #         class_inds = bboxes_prd[..., 5].astype(np.int32)
        #         scores = bboxes_prd[..., 4]

        #         visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.classes)
        #         path = os.path.join(cfg.PROJECT_PATH, "data/results/{}.jpg".format(self.__visual_imgs))
        #         cv2.imwrite(path, img)

        #         self.__visual_imgs += 1

        #     for bbox in bboxes_prd:
        #         coor = np.array(bbox[:4], dtype=np.int32)
        #         score = bbox[4]
        #         class_ind = int(bbox[5])

        #         class_name = self.classes[class_ind]
        #         score = '%.4f' % score
        #         xmin, ymin, xmax, ymax = map(str, coor)
        #         s = ' '.join([img_ind, score, xmin, ymin, xmax, ymax]) + '\n'

        #         with open(os.path.join(self.pred_result_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
        #             f.write(s)

        return self.__calc_APs()

    def get_bbox(self, img, multi_test=False, flip_test=False):
        if multi_test:
            test_input_sizes = range(320, 640, 96)
            bboxes_list = []
            for test_input_size in test_input_sizes:
                valid_scale =(0, np.inf)
                bboxes_list.append(self.__predict(img, test_input_size, valid_scale))
                if flip_test:
                    bboxes_flip = self.__predict(img[:, ::-1], test_input_size, valid_scale)
                    bboxes_flip[:, [0, 2]] = img.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
        else:
            bboxes = self.__predict(img, self.val_shape, (0, np.inf))

        bboxes = nms(bboxes, self.conf_thresh, self.nms_thresh)

        return bboxes

    def __predict(self, img, test_shape, valid_scale):
        org_img = np.copy(img)
        org_h, org_w, _ = org_img.shape

        img = self.__get_img_tensor(img, test_shape).to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, p_d = self.model(img)
        pred_bbox = p_d.squeeze().cpu().numpy()
        bboxes = self.__convert_pred(pred_bbox, test_shape, (org_h, org_w), valid_scale)

        return bboxes

    def __get_img_tensor(self, img, test_shape):
        img = Resize((test_shape, test_shape), correct_box=False)(img, None).transpose(2, 0, 1)
        return torch.from_numpy(img[np.newaxis, ...]).float()


    def __convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
        """
        預測框進行過濾，去除尺度不合理的框
        """
        pred_coor = xywh2xyxy(pred_bbox[:, :4])
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # (1)
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        # 需要注意的是，無論我們在訓練的時候使用什麼數據增強方式，都不影響此處的轉換方式
        # 假設我們對輸入測試圖片使用了轉換方式A，那麼此處對bbox的轉換方式就是方式A的逆向過程
        org_h, org_w = org_img_shape
        resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
        dw = (test_input_size - resize_ratio * org_w) / 2
        dh = (test_input_size - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # (2)將預測的bbox中超出原圖的部分裁掉
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        # (3)將無效bbox的coor置為0
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # (4)去掉不在有效範圍內的bbox
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # (5)將score低於score_threshold的bbox去掉
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.conf_thresh

        mask = np.logical_and(scale_mask, score_mask)

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

        return bboxes


    def __calc_APs(self, iou_thresh=0.5, use_07_metric=False):
        """
        計算每個類別的ap值
        :param iou_thresh:
        :param use_07_metric:
        :return:dict{cls:ap}
        """
        filename = os.path.join(self.pred_result_path, 'comp4_det_test_{:s}.txt')
        cachedir = os.path.join(self.pred_result_path, 'cache')
        annopath = os.path.join(self.val_data_path_07, 'Annotations', '{:s}.xml')
        imagesetfile = os.path.join(self.val_data_path_07,  'ImageSets', 'Main', 'val.txt')
        
        annopath2 = os.path.join(self.val_data_path_12, 'Annotations', '{:s}.xml')
        imagesetfile2 = os.path.join(self.val_data_path_12,  'ImageSets', 'Main', 'val.txt')
        APs = {}
        for i, cls in enumerate(self.classes):
            R, P, AP = voc_eval(filename, annopath, imagesetfile, annopath2, imagesetfile2, cls, cachedir, iou_thresh, use_07_metric)
            APs[cls] = AP
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)

        return APs