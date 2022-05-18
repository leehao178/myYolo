import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import cv2
import random
from models.utils.bbox import xyxy2xywh


def letterbox(img, new_shape=416, color=(127.5, 127.5, 127.5), mode='square'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    ratiow, ratioh = ratio, ratio
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
    elif mode is 'scaleFill':
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape, new_shape)
        ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratiow, ratioh, dw, dh


class ImageHSV(object):
    def __init__(self, fraction=0.5):
        # SV augmentation by 50%, must be < 1.0
        self.fraction = fraction

    def __call__(self, data):
        img, bboxes = data
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
        S = img_hsv[:, :, 1].astype(np.float32)  # saturation
        V = img_hsv[:, :, 2].astype(np.float32)  # value

        a = (random.random() * 2 - 1) * self.fraction + 1
        b = (random.random() * 2 - 1) * self.fraction + 1
        S *= a
        V *= b

        img_hsv[:, :, 1] = S if a < 1 else S.clip(None, 255)
        img_hsv[:, :, 2] = V if b < 1 else V.clip(None, 255)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return img, bboxes


class RandomHorizontalFilp(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        # bboxes = xyxy
        img, bboxes = data
        if len(bboxes) > 0 and random.random() < self.prob:
            _, w, _ = img.shape
            img = np.fliplr(img)
            # bboxes[:, [1, 3]] = w - bboxes[:, [3, 1]]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        return img, bboxes

# TODO have bugs
class RandomCrop(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        # bboxes = xyxy
        img, bboxes = data
        if len(bboxes) > 0 and random.random() < self.p:
            h_img, w_img, _ = img.shape

            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            # max_bbox = np.concatenate([np.min(bboxes[:, 1:3], axis=0), np.max(bboxes[:, 3:5], axis=0)], axis=-1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            img = img[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            # bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_xmin
            # bboxes[:, [2, 4]] = bboxes[:, [2, 4]] - crop_ymin
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        return img, bboxes


class RandomAffine(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        # bboxes = xyxy
        img, bboxes = data
        if len(bboxes) > 0 and random.random() < self.p:
            h_img, w_img, _ = img.shape
            # 得到可以包含所有bbox的最大bbox
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            # max_bbox = np.concatenate([np.min(bboxes[:, 1:3], axis=0), np.max(bboxes[:, 3:5], axis=0)], axis=-1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w_img, h_img))

            # bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + tx
            # bboxes[:, [2, 4]] = bboxes[:, [2, 4]] + ty
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
        return img, bboxes


class Pad(object):
    def __init__(self, img_size=416):
        self.img_size = img_size
    def __call__(self, data):
        # bboxes = xyxy
        img, bboxes = data
        h, w, _ = img.shape
        shape = self.img_size
        img, ratiow, ratioh, padw, padh = letterbox(img, new_shape=shape, mode='square')
        labels = bboxes.copy()
        # labels[:, 1] = ratiow * w * (bboxes[:, 1] - bboxes[:, 3] / 2) + padw
        # labels[:, 2] = ratioh * h * (bboxes[:, 2] - bboxes[:, 4] / 2) + padh
        # labels[:, 3] = ratiow * w * (bboxes[:, 1] + bboxes[:, 3] / 2) + padw
        # labels[:, 4] = ratioh * h * (bboxes[:, 2] + bboxes[:, 4] / 2) + padh

        labels[:, [1, 3]] = bboxes[:, [1, 3]] * ratiow + padw
        labels[:, [2, 4]] = bboxes[:, [2, 4]] * ratioh + padh
        
        labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
        
        return img, labels


class Resize(object):
    """
    RGB轉換 ----->  resize(保持原有長寬比)
    並可以選擇是否校正bbox
    """
    def __init__(self, target_shape, correct_box=True):
        self.h_target, self.w_target = target_shape
        self.correct_box = correct_box

    def __call__(self, data):
        img, bboxes = data
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

class RelativeLabels(object):
    # 相對標籤
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, bboxes = data
        h, w, _ = img.shape
        bboxes[:, [1, 3]] /= w
        bboxes[:, [2, 4]] /= h
        return img, bboxes


class Normalize(object):
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
    def __call__(self, data):
        img, boxes = data
        img = img.copy().astype(np.float32)
        assert img.dtype != np.uint8
        mean = np.float64(self.mean.reshape(1, -1))
        stdinv = 1 / np.float64(self.std.reshape(1, -1))
        if self.to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        return img, boxes


class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data

        # Extract image as PyTorch tensor
        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = torch.from_numpy(img)

        bb_targets = torch.zeros((len(boxes), 6))
        if len(boxes):
            bb_targets[:, 1:] = torch.from_numpy(boxes)
        return img, bb_targets
    

class AbsoluteLabels(object):
    # 絕對標籤
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, bboxes = data
        # print(img.shape)
        h, w, _ = img.shape
        bboxes[:, [1, 3]] *= w
        bboxes[:, [2, 4]] *= h
        return img, bboxes