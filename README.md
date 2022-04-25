# Yolo Detection

# install
```
 conda create --name "yolo" python=3.7 -y

 conda activate yolo

 pip install -r requirements.txt

 pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

---
## Results

| name | train size | test size | Train Dataset | Val Dataset | mAP | notes |
|:-----|   :-----   |  :------  |   :------     |   :------   |:-----|:-----|
|YOLOv3|     416    |    416    | 2007+2012 trainval | 2007test | 66.37 | baseline |
|YOLOv3|     416    |    512    | 2007+2012 trainval | 2007test | 67.92 | baseline |
|YOLOv3|     544    |    544    | 2007+2012 trainval | 2007test | 70.24 | baseline |



---
## Environment

* Nvida GeForce RTX 1080 Ti
* CUDA11.1
* ubuntu 18.04
* python 3.7
