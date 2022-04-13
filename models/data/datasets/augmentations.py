from torchvision import transforms
import sys,os
sys.path.append(os.getcwd())
from models.data.datasets.transforms import ToTensor, RandomHorizontalFilp
from models.data.datasets.transforms import Normalize



AUGMENTATION_TRANSFORMS = transforms.Compose([
    RandomHorizontalFilp(),
    # Normalize(mean=[123.675, 116.28, 103.53],
    #           std=[58.395, 57.12, 57.375],
    #           to_rgb=True),
    ToTensor(),
])