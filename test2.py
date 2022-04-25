import os
import xml.etree.ElementTree as ET
import cv2
list_path ='/home/lab602.demo/.pipeline/datasets/VOCdevkit'

# with open(os.path.join(list_path, 'VOC2007', 'ImageSets', 'Main', 'val.txt'), "r") as f:
#     img_ids = f.readlines()

imgs_path = '/home/lab602.demo/.pipeline/10678031/myYolo/outputs/voc/results/img'
det_imgs = os.listdir('/home/lab602.demo/.pipeline/10678031/myYolo/outputs/voc/results/img')


for img_name in det_imgs:
    img_id = img_name.split('.')[0]
    xml_path = os.path.join(list_path, 'VOC2007', 'Annotations', f'{img_id.strip()}.xml')
    img_path = os.path.join(imgs_path, img_name)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []

    target_img_size = root.find('size')
    width = int(target_img_size.find('width').text)
    height = int(target_img_size.find('height').text)
    img = cv2.imread(img_path)
    for obj in root.findall('object'):
        cls_name = obj.find('name').text

        difficult = obj.find("difficult").text.strip()
        # difficult 表示是否容易識別，0表示容易，1表示困難
        if (not True) and (int(difficult) == 1):
            continue
        bnd_box = obj.find('bndbox')
        # TODO: check whether it is necessary to use int
        # Coordinates may be float type
        xmin = int(float(bnd_box.find('xmin').text))
        ymin = int(float(bnd_box.find('ymin').text))
        xmax = int(float(bnd_box.find('xmax').text))
        ymax = int(float(bnd_box.find('ymax').text))

        if int(difficult) == 1:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        text = '{}'.format(cls_name)
        txt_color = color
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        if int(difficult) == 1:
            txt_bk_color = (0, 0, 255*0.7)
        else:
            txt_bk_color = (0, 255*0.7, 0)

        cv2.rectangle(
            img,
            (int(xmin), int(ymin) + 1),
            (int(xmin) + txt_size[0] + 1, int(ymin) + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (int(xmin), int(ymin) + txt_size[1]), font, 0.4, txt_color, thickness=1)

    cv2.imwrite('./outputs/voc/results/img2/{}.jpg'.format(img_id), img)
