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
from models.utils.bbox import xyxy2xywh, resize


class ListDataset(Dataset):
    def __init__(self,
                 list_path,
                 img_size=416,
                 multiscale=True,
                 transform=None,
                 years={'VOC2007':'trainval', 'VOC2012': 'trainval'},
                 use_difficult_bbox=True):
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
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.img_files = []
        self.imgs_ids = []
        self.label_files = []
        
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
                    # difficult 表示是否容易識別，0表示容易，1表示困難
                    if (not use_difficult_bbox) and (int(difficult) == 1):
                        continue
                    bnd_box = obj.find('bndbox')
                    # TODO: check whether it is necessary to use int
                    # Coordinates may be float type
                    xmin = float(bnd_box.find('xmin').text)
                    ymin = float(bnd_box.find('ymin').text)
                    xmax = float(bnd_box.find('xmax').text)
                    ymax = float(bnd_box.find('ymax').text)

                    # bbox = [
                    #     label,
                    #     x/width,
                    #     y/height,
                    #     w/width,
                    #     h/height,
                    # ]

                    bbox = [
                        label,
                        xmin,
                        ymin,
                        xmax,
                        ymax,
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

        return img_id, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1
        # print(self.batch_count)
        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))
        
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)
        return imgs, bb_targets, paths

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