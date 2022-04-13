import os
import shutil
from models.eval.voc_eval import voc_eval
import cv2
import numpy as np
import torch
from tqdm import tqdm
from models.utils.visualization import _COLORS
from models.data.datasets.transforms import letterbox
from models.utils.nms import non_max_suppression


class Evaluator(object):
    def __init__(self, model, dataloader, visiual=True):
        self.classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
        self.pred_result_path = os.path.join('/home/lab602.demo/.pipeline/10678031/myYolo/outputs/voc', 'results')
        self.val_data_path_07 = os.path.join('/home/lab602.demo/.pipeline/datasets/VOCdevkit', 'VOC2007')
        self.val_data_path_12 = os.path.join('/home/lab602.demo/.pipeline/datasets/VOCdevkit', 'VOC2012')
        self.conf_thresh = 0.5
        self.nms_thresh = 0.5
        self.val_shape =  416

        self.dataloader = dataloader

        self.model = model
        self.device = next(model.parameters()).device

    def APs_voc(self, multi_test=False, flip_test=False, save_json=False):
        self.model.eval()
        nc = len(self.classes)  # number of classes
        conf_thres = 0.3
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
            # print(img_ids)
            # Statistics per image
            for si, pred in enumerate(output):
                img_path = os.path.join(self.val_data_path_07, 'JPEGImages', f'{img_ids[si]}.jpg')
                img = cv2.imread(img_path)

                h, w, _ = img.shape
                shape = 416
                padding_img, ratiow, ratioh, padw, padh = letterbox(img, new_shape=shape, mode='square')
                

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
                    xmin = ((xmin - padw) / ratiow) / w
                    ymin = ((ymin - padh) / ratioh) / h
                    xmax = ((xmax - padw) / ratiow) / w
                    ymax = ((ymax - padh) / ratioh) / h


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
        return self.calc_APs()


    def calc_APs(self, iou_thresh=0.5, use_07_metric=False):
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