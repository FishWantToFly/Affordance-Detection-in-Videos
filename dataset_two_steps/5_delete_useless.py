'''
2020.4.28
Do data augmentation on dataset_original 
Use 
    1. horizontally flip 
		Do on dataset_original only
		Need to copy : 
			first_mask -> flip
			inpaint_depth -> flip
			raw_frames -> flip

No need to do with shift / occlusion (do it when training)
'''

import glob, os, copy, random
import numpy as np
from os import walk
from PIL import Image, ImageOps, ImageEnhance

def create_dir(new_dir):
	if not os.path.exists(new_dir):
		os.mkdir(new_dir)

depth_dir = 'raw_depth'
frame_dir = 'raw_frames'
mask_dir = 'first_mask'
depth_dir = 'inpaint_depth'
now_dir = os.getcwd()
IMG_HEIGHT = 480
IMG_WIDTH = 640


att_heatmap_dir = 'attention_heatmap'
first_mask_dir = 'first_mask'
first_mask_rgb_dir = 'first_mask_rgb'




######################################################
# Step 1 : delete three directory
# test_action = "/home/s5078345/Affordance-Detection-on-Video/dataset/dataset_original/lab/chair_1/put_big_object_on_it"

for action in sorted(glob.glob("./dataset_original/*/*/*")):
	_, _action = os.path.split(action)

	object_semantic = action.split('/')[3].split('_')[0]
	target_list = ['chair', 'table']
	if object_semantic not in target_list:
		continue

	print(action)

	# 1. create directory
	place = action.split('/')[-3]
	_object = action.split('/')[-2]

	att_heatmap_path = os.path.join(action, att_heatmap_dir)
	first_mask_path = os.path.join(action, first_mask_dir)
	first_mask_rgb_path = os.path.join(action, first_mask_rgb_dir)

	if os.path.exists(att_heatmap_path):
		os.system("rm -rf %s" % (att_heatmap_path))
	if os.path.exists(first_mask_path):
		os.system("rm -rf %s" % (first_mask_path))
	if os.path.exists(first_mask_rgb_path):
		os.system("rm -rf %s" % (first_mask_rgb_path))


	# break