import os
import shutil
from models.eval.voc_eval import voc_eval
import cv2
import numpy as np
import torch
from tqdm import tqdm
from models.utils.visualization import _COLORS
from models.data.datasets.transforms import letterbox
from models.utils.nms import non_max_suppression, nms
from models.utils.visualization import visualize_boxes
import random
from models.data.datasets.transforms import Resize
from models.utils.bbox import xywh2xyxy


class Evaluator(object):
    def __init__(self, model, visiual=True):
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
          'bus', 'car', 'cat', 'chair', 'cow',
          'diningtable', 'dog', 'horse', 'motorbike', 'person',
          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.pred_result_path = os.path.join('/home/lab602.demo/.pipeline/10678031/myYolo/outputs', 'voc', 'results')
        self.val_data_path = os.path.join('/home/lab602.demo/.pipeline/datasets', 'VOCdevkit', 'VOC2007')
        self.conf_thresh = 0.01
        self.nms_thresh = 0.5
        self.val_shape = 544

        self.__visiual = visiual
        self.__visual_imgs = 0

        self.model = model
        self.device = next(model.parameters()).device

    def APs_voc(self, multi_test=False, flip_test=False):
        img_inds_file = os.path.join(self.val_data_path,  'ImageSets', 'Main', 'test.txt')
        with open(img_inds_file, 'r') as f:
            lines = f.readlines()
            img_inds = [line.strip() for line in lines]

        if os.path.exists(self.pred_result_path):
            shutil.rmtree(self.pred_result_path)
        os.mkdir(self.pred_result_path)

        for img_ind in tqdm(img_inds):
            img_path = os.path.join(self.val_data_path, 'JPEGImages', img_ind+'.jpg')
            img = cv2.imread(img_path)
            bboxes_prd = self.get_bbox(img, multi_test, flip_test)

            if bboxes_prd.shape[0]!=0 and self.__visiual and self.__visual_imgs < 100:
                boxes = bboxes_prd[..., :4]
                class_inds = bboxes_prd[..., 5].astype(np.int32)
                scores = bboxes_prd[..., 4]

                visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.classes)
                path = os.path.join('/home/lab602.demo/.pipeline/10678031/myYolo', "outputs/voc/{}.jpg".format(self.__visual_imgs))
                cv2.imwrite(path, img)
                # print("saving predict images : {}".format(path))

                self.__visual_imgs += 1

            for bbox in bboxes_prd:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])

                class_name = self.classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = map(str, coor)
                s = ' '.join([img_ind, score, xmin, ymin, xmax, ymax]) + '\n'

                with open(os.path.join(self.pred_result_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
                    f.write(s)


        return self.calc_APs()

    def get_bbox(self, img, multi_test=False, flip_test=False):
        if multi_test:
            test_input_sizes = random.choice(range(10, 20, 3)) * 32
            bboxes_list = []
            for test_input_size in test_input_sizes:
                valid_scale =(90, np.inf)
                bboxes_list.append(self.predict(img, test_input_size, valid_scale))
                if flip_test:
                    bboxes_flip = self.predict(img[:, ::-1], test_input_size, valid_scale)
                    bboxes_flip[:, [0, 2]] = img.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
        else:
            bboxes = self.predict(img, self.val_shape, (0, np.inf))

        bboxes = nms(bboxes, self.conf_thresh, self.nms_thresh)

        return bboxes

    def predict(self, img, test_shape, valid_scale):
        org_img = np.copy(img)
        org_h, org_w, _ = org_img.shape

        img = self.get_img_tensor(img, test_shape).to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, p_d = self.model(img)
        pred_bbox = p_d.squeeze().cpu().numpy()
        bboxes = self.convert_pred(pred_bbox, test_shape, (org_h, org_w), valid_scale)

        return bboxes

    def get_img_tensor(self, img, test_shape):
        img = Resize((test_shape, test_shape), correct_box=False)((img, None)).transpose(2, 0, 1)
        return torch.from_numpy(img[np.newaxis, ...]).float()


    def convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
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


    def calc_APs(self, iou_thresh=0.5, use_07_metric=False):
        """
        计算每个类别的ap值
        :param iou_thresh:
        :param use_07_metric:
        :return:dict{cls:ap}
        """
        filename = os.path.join(self.pred_result_path, 'comp4_det_test_{:s}.txt')
        cachedir = os.path.join(self.pred_result_path, 'cache')
        annopath = os.path.join(self.val_data_path, 'Annotations', '{:s}.xml')
        imagesetfile = os.path.join(self.val_data_path,  'ImageSets', 'Main', 'test.txt')
        APs = {}
        for i, cls in enumerate(self.classes):
            R, P, AP = voc_eval(filename, annopath, imagesetfile, cls, cachedir, iou_thresh, use_07_metric)
            APs[cls] = AP
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)

        return APs