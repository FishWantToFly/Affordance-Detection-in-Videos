'''
2020.6.5 step 1 for coco dataset 
just provide one image per action
'''

from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math
import glob

import torch
import torch.utils.data as data

import sys
sys.path.append('..')

from utils.osutils import *
from utils.imutils import *
from utils.transforms import *

img_path = '/home/s5078345/Affordance-Detection-on-Video/dataset_coco/dataset_horizontally_flip/coco_all/train2017_with_binary_mask/image_1259/raw_frames/00000.png'


img = load_image(img_path)
print(img)
print(img.shape)