# This Python file uses the following encoding: utf-8

# if__name__ == "__main__":
#     pass
from he_wenjian.python_1_he import he
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloaders import custom_transforms as trforms
from dataloaders import tn3k
from dataloaders import utils
from dataloaders.utils import get_dice
from dataloaders.utils import cal_HD
from dataloaders.utils import cal_HD_2
from Unet_bot_Transformer.Unet import UNet
def pytorch_test():
        a = torch.rand(2)
        x = np.array(a)
        print(x)
        he()
