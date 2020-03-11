'''
Report statistics for dataset 
1. how many different action
2. how many action
3. average / maximum / minimum frames
'''

import glob, os, copy, re
import numpy as np
from os import walk
from PIL import Image
import statistics 

######################################################
# Step 1 : visualize from raw depth information (transform to greyscale image)
test_action = "/home/s5078345/affordance/dataset/lab/chair_1/move_object_with_thing_in_it/"
depth_dir = 'raw_depth'
frame_dir = 'raw_frames'
mask_dir = 'mask'

semantic_dict = {'basket': 0, 'chair': 1, 'plate': 2,'sofa': 3, 'table': 4}
for action in glob.glob("../dataset/dataset_original/*/*/*"):	
	semantic = os.path.basename(os.path.dirname(action)).split('_')[0]
	semantic_dict[semantic] += 1

print(semantic_dict)