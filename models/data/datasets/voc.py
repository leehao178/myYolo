from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import os
import warnings
import numpy as np
import os.path as osp
import xml.etree.ElementTree as ET
import cv2
from models.data.datasets.transforms import letterbox
from models.utils.bbox import xyxy2xywh, resize, bbox_iou_np


class ListDataset(Dataset):
    def __init__(self,
                 list_path,
                 img_size=416,
                 multiscale=True,
                 transform=None,
                 years={'VOC2007':'trainval', 'VOC2012': 'trainval'},
                 use_difficult_bbox=True,
                 configs=None):
        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform
        self.CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
        self.num_classes = 20
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.img_files = []
        self.imgs_ids = []
        self.label_files = []
        self.configs = configs

        self.load_annotations(years=years, list_path=list_path, use_difficult_bbox=use_difficult_bbox)


    def load_annotations(self, years, list_path, use_difficult_bbox=False):
        for year, train_type in years.items():
            with open(os.path.join(list_path, year, 'ImageSets', 'Main', '{}.txt'.format(train_type)), "r") as f:
                    img_ids = f.readlines()
            for img_id in img_ids:
                self.img_files.append(os.path.join(list_path, year, 'JPEGImages', f'{img_id.strip()}.jpg'))
                self.imgs_ids.append(img_id.strip())
                xml_path = os.path.join(list_path, year, 'Annotations', f'{img_id.strip()}.xml')
                tree = ET.parse(xml_path)
                root = tree.getroot()
                bboxes = []

                target_img_size = root.find('size')
                width = int(target_img_size.find('width').text)
                height = int(target_img_size.find('height').text)
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name not in self.CLASSES:
                        continue
                    label = self.cat2label[name]

                    difficult = obj.find("difficult").text.strip()
                    # difficult ???????????????????????????0???????????????1????????????
                    if (not use_difficult_bbox) and (int(difficult) == 1):
                        continue
                    bnd_box = obj.find('bndbox')
                    # TODO: check whether it is necessary to use int
                    # Coordinates may be float type
                    xmin = float(bnd_box.find('xmin').text)
                    ymin = float(bnd_box.find('ymin').text)
                    xmax = float(bnd_box.find('xmax').text)
                    ymax = float(bnd_box.find('ymax').text)

                    bbox = [
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                        label,
                    ]
                    bboxes.append(bbox)
                bboxes = np.array(bboxes)
                self.label_files.append(bboxes.astype(np.float32))  

    def __getitem__(self, index):
        # ---------Images---------
        try:
            img_path = self.img_files[index]
            img_id = self.imgs_ids[index]
            # img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
            img = cv2.imread(img_path)  # H*W*C and C=BGR
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # TODO warning empty labels
        # ---------Labels---------
        try:
            boxes = self.label_files[index]
            # print(boxes)
        except Exception:
            # print(f"Could not read label '{label_path}'.")
            return

        # -------Transforms-------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print(Exception)
                print("Could not apply transform.")
                return

        img = img.transpose(2, 0, 1)

        bboxes_xywh, label = self.creat_label(bboxes=bb_targets,
                                              anchors=np.array(self.configs['anchors']),
                                              strides=np.array(self.configs['strides'][0]))

        img = torch.from_numpy(img).float()


        
        bboxes_xywh = [torch.from_numpy(bbox).float() for bbox in bboxes_xywh]
        label = [torch.from_numpy(labels).float() for labels in label]


        return img, bboxes_xywh, label
    
    def creat_label(self, bboxes, anchors, strides):
        """
        ????????????.???????????????????????????GT???bboxes??????anchor.
        (1)??????????????????bbox,??????????????????xyxy2xywh???????????????????????????????????????bbox xywh????????????
        (2)?????????bbox?????????????????????anchors??????iou??????????????????iou?????????anchor????????????bbox???
        ???????????????????????????iou???????????????????????????????????????????????????iou?????????anchor??????????????????

        ?????????1????????????GT????????????????????????anchor,??????anchor???????????????????????????????????????????????????
            2???bbox???????????????????????????????????????????????????GT????????????????????????????????????
        :param img: ?????????????????????????????????[0-1]
        :param bboxes: bboxes??????????????????GT????????????[N, 5]?????? ???????????????[xmin, ymin, xmax, ymax, cls]
        :return: img, label
        """

        # TRAIN_OUTPUT_SIZE = self.img_size / strides
        feature_sizes = self.img_size / strides
        # ANCHOR_PER_SCALE = 3
        num_anchors = 3

        label = [np.zeros((int(feature_sizes[i]), int(feature_sizes[i]), num_anchors, 5+self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((150, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])

            # onehot
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0

            # ???xyxy ??????xywh
            # bbox_xywh = tools.xyxy2xywh(bbox_coor.reshape(1, -1))
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)

            # print("bbox_xywh: ", bbox_xywh)

            # ?????????????????????????????????xywh
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((num_anchors, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # 0.5 ????????????
                anchors_xywh[:, 2:4] = anchors[i] / strides[i].repeat(2)
                # anchors_xywh[:, 2:4] = ANCHORS[i]

                iou_scale = bbox_iou_np(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    # Bug : ?????????bbox???????????????anchor??????????????????anchor?????????????????????bbox
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = one_hot

                    bbox_ind = int(bbox_count[i] % 150)  # BUG : 150??????????????????
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)  # ??????????????????iou
                best_detect = int(best_anchor_ind / num_anchors)
                best_anchor = int(best_anchor_ind % num_anchors)

                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = one_hot

                bbox_ind = int(bbox_count[best_detect] % 150)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        return bboxes_xywh, label

    # def collate_fn(self, batch):
    #     self.batch_count += 1
    #     # print(self.batch_count)
    #     # Drop invalid images
    #     batch = [data for data in batch if data is not None]

    #     paths, imgs, bb_targets = list(zip(*batch))

    #     # Selects new image size every tenth batch
    #     if self.multiscale and self.batch_count % 10 == 0:
    #         self.img_size = random.choice(
    #             range(self.min_size, self.max_size + 1, 32))
        
    #     # Resize images to input shape
    #     imgs = torch.stack([resize(img, self.img_size) for img in imgs])

    #     # Add sample index to targets
    #     for i, boxes in enumerate(bb_targets):
    #         boxes[:, 0] = i
    #     bb_targets = torch.cat(bb_targets, 0)
    #     return imgs, bb_targets, paths

    def __len__(self):
        return len(self.img_files)
    

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from augmentations import AUGMENTATION_TRANSFORMS
    from tqdm import tqdm
    dataset = ListDataset(
        '/home/lab602.demo/.pipeline/datasets/VOCdevkit',
        img_size=416,
        multiscale=False,
        transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=dataset.collate_fn,)
        # worker_init_fn=worker_seed_set)
    # return dataloader

    for i in tqdm(dataloader):
        print(len(i))