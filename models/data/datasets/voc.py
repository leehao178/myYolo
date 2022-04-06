from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import random
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile
import os.path as osp
import xml.etree.ElementTree as ET

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)

class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None, years=['VOC2007', 'VOC2012'], traintype='train'):
        self.CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.img_files = []
        self.imgs_ids = []
        self.label_files = []
        for year in years:
            with open(os.path.join(list_path, year, 'ImageSets', 'Main', '{}.txt'.format(traintype)), "r") as f:
                img_ids = f.readlines()
            for img_id in img_ids:
                self.img_files.append(os.path.join(list_path, year, 'JPEGImages', f'{img_id.strip()}.jpg'))
                self.imgs_ids.append(img_id.strip())
                xml_path = os.path.join(list_path, year, 'Annotations', f'{img_id.strip()}.xml')
                tree = ET.parse(xml_path)
                root = tree.getroot()
                bboxes = []
                labels = []
                bboxes_ignore = []
                labels_ignore = []
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name not in self.CLASSES:
                        continue
                    label = self.cat2label[name]
                    difficult = obj.find('difficult')
                    difficult = 0 if difficult is None else int(difficult.text)
                    bnd_box = obj.find('bndbox')
                    # TODO: check whether it is necessary to use int
                    # Coordinates may be float type
                    xmin = int(float(bnd_box.find('xmin').text))
                    ymin = int(float(bnd_box.find('ymin').text))
                    xmax = int(float(bnd_box.find('xmax').text))
                    ymax = int(float(bnd_box.find('ymax').text))
                    w = xmax - xmin
                    h = ymax - ymin
                    x = xmin + w*0.5
                    y = ymin + h*0.5
                    bbox = [
                        label,
                        x,
                        y,
                        w,
                        h,
                        
                    ]
                    bboxes.append(bbox)
                bboxes = np.array(bboxes)
                self.label_files.append(bboxes.astype(np.float32))
                # print('========')
                # print(len(self.label_files))
                #     ignore = False
                #     if self.min_size:
                #         assert not self.test_mode
                #         w = bbox[2] - bbox[0]
                #         h = bbox[3] - bbox[1]
                #         if w < self.min_size or h < self.min_size:
                #             ignore = True
                #     if difficult or ignore:
                #         bboxes_ignore.append(bbox)
                #         labels_ignore.append(label)
                #     else:
                #         bboxes.append(bbox)
                #         labels.append(label)
                # if not bboxes:
                #     bboxes = np.zeros((0, 4))
                #     labels = np.zeros((0, ))
                # else:
                #     bboxes = np.array(bboxes, ndmin=2) - 1
                #     labels = np.array(labels)
                # if not bboxes_ignore:
                #     bboxes_ignore = np.zeros((0, 4))
                #     labels_ignore = np.zeros((0, ))
                # else:
                #     bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
                #     labels_ignore = np.array(labels_ignore)
                # ann = dict(
                #     bboxes=bboxes.astype(np.float32),
                #     labels=labels.astype(np.int64),
                #     bboxes_ignore=bboxes_ignore.astype(np.float32),
                #     labels_ignore=labels_ignore.astype(np.int64))
                # return ann

        # with open(list_path, "r") as file:
        #     self.img_files = file.readlines()

        # self.label_files = []
        # for path in self.img_files:
        #     image_dir = os.path.dirname(path)
        #     label_dir = "labels".join(image_dir.rsplit("images", 1))
        #     assert label_dir != image_dir, \
        #         f"Image path must contain a folder named 'images'! \n'{image_dir}'"
        #     label_file = os.path.join(label_dir, os.path.basename(path))
        #     label_file = os.path.splitext(label_file)[0] + '.txt'
        #     self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):
        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index]
            img_id = self.imgs_ids[index]
            # print('=================img_path====================')
            # print(index)
            # print(img_path)

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            '''
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
            '''
            boxes = self.label_files[index]
        except Exception:
            # print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                # print('-----------------')
                # print(img.shape)
                # print(boxes.shape)
                # print(boxes)

                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return
        # print(img_id)
        return img_id, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

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

        # print('aegsdfgasfg')
        # print(bb_targets.shape)
        # print(bb_targets)

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

