import os
import shutil
from sys import path
from models.eval.voc_eval import voc_eval
import cv2
import numpy as np
import torch
from tqdm import tqdm
from models.utils.visualization import _COLORS
from models.data.datasets.transforms import letterbox
from models.utils.nms import non_max_suppression


class Evaluator(object):
    def __init__(self, model, dataloader, configs, epoch='test'):
        self.classes = configs['classes']
        self.outputs_path = '/home/lab602.demo/.pipeline/10678031/myYolo/outputs/voc'
        self.pred_result_path = os.path.join('/home/lab602.demo/.pipeline/10678031/myYolo/outputs/voc', 'results')
        self.val_data_path_07 = os.path.join('/home/lab602.demo/.pipeline/datasets/VOCdevkit', 'VOC2007')
        self.conf_thresh = configs['conf_thres']
        self.nms_thresh = configs['nms_thres']
        self.test_size =  configs['test_size']
        self.dataloader = dataloader
        self.model = model
        self.device = next(model.parameters()).device
        self.epoch = epoch


    def APs_voc(self, multi_test=False, flip_test=False, save_json=False):
        self.model.eval()
        nc = len(self.classes)  # number of classes
        # img_inds_file = os.path.join(self.val_data_path,  'ImageSets', 'Main', 'test.txt')
        # with open(img_inds_file, 'r') as f:
        #     lines = f.readlines()
        #     img_inds = [line.strip() for line in lines]

        self.pred_result_path = os.path.join(self.outputs_path, 'results_{}'.format(self.epoch))
        
        if os.path.isdir(self.pred_result_path) == False:
            os.mkdir(self.pred_result_path)
        if os.path.isdir(os.path.join(self.pred_result_path, 'img')) == False:
            os.mkdir(os.path.join(self.pred_result_path, 'img'))
        # print(('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1'))

        # for class_name in self.classes:
        #     with open(os.path.join(self.pred_result_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
        #         pass

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
            



            
            # print(inf_out.shape)
            # torch.Size([64, 18207, 25])
            # Run NMS
            output = non_max_suppression(inf_out, conf_thres=self.conf_thresh, nms_thres=self.nms_thresh)
            # print(output[0].shapse)
            # torch.Size([3, 7])
            # Statistics per image
            for si, preds in enumerate(output):

                img_path = os.path.join(self.val_data_path_07, 'JPEGImages', f'{img_ids[si]}.jpg')

                img = cv2.imread(img_path)

                h, w, _ = img.shape
                shape = self.test_size
                padding_img, ratiow, ratioh, padw, padh = letterbox(img, new_shape=shape, mode='square')

                if preds is None:
                    continue
                
                for pred in preds:
                    xmin, ymin, xmax, ymax, conf, cls_score, cls = pred
                    xmin = (xmin - padw) / ratiow
                    ymin = (ymin - padh) / ratioh
                    xmax = (xmax - padw) / ratiow
                    ymax = (ymax - padh) / ratioh

                    if xmin < 0:
                        xmin = 0
                    if ymin < 0:
                        ymin = 0
                    
                    if xmax > w:
                        xmax = w
                    if ymax > h:
                        ymax = h

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
                    conf = '%.4f' % conf
                    s = ' '.join([str(img_ids[si]), conf, str(float(xmin)), str(float(ymin)), str(float(xmax)), str(float(ymax))]) + '\n'

                    with open(os.path.join(self.pred_result_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
                        f.write(s)
                cv2.imwrite(os.path.join(self.pred_result_path, 'img', '{}.jpg'.format(img_ids[si])), img)
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
        imagesetfile = os.path.join(self.val_data_path_07,  'ImageSets', 'Main', 'test.txt')

        APs = {}
        for i, cls in enumerate(self.classes):
            R, P, AP = voc_eval(filename, annopath, imagesetfile, cls, cachedir, iou_thresh, use_07_metric)
            APs[cls] = AP
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)

        return APs