'''
2020.6.5
Do data augmentation on dataset_original 
Use 
    1. horizontally flip 
		Do on dataset_original only
		Need to copy : 
			mask -> flip
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

# depth_dir = 'raw_depth'
frame_dir = 'raw_frames'
mask_dir = 'mask'
# depth_dir = 'inpaint_depth'
now_dir = os.getcwd()
IMG_HEIGHT = 480
IMG_WIDTH = 640


######################################################
# Step 1 : horizontally flip 
copy_dir_name = 'dataset_horizontally_flip'
if not os.path.exists(copy_dir_name):
	os.mkdir(copy_dir_name)

# for action in sorted(glob.glob("./dataset_original/coco_all/*/*")):
for action in sorted(glob.glob("./dataset_original/*/*/*")):
	_, _action = os.path.split(action)
	object_semantic = action.split('/')[3].split('_')[0]
	print(action)

	# 1. create directory
	place = action.split('/')[-3]
	_object = action.split('/')[-2]


	new_place_dir = os.path.join(now_dir, copy_dir_name, place)
	new_object_dir = os.path.join(now_dir, copy_dir_name, place, _object)
	if not os.path.exists(new_place_dir):
		os.mkdir(new_place_dir)
	if not os.path.exists(new_object_dir):
		os.mkdir(new_object_dir)

	action_mask_path = os.path.join(action, mask_dir, '*')
	action_frame_path = os.path.join(action, frame_dir, '*')
	mask_list = sorted(glob.glob(action_mask_path))
	frame_list = sorted(glob.glob(action_frame_path))

	new_action_path = os.path.join(now_dir, copy_dir_name, place, _object, _action)
	create_dir(new_action_path)
	new_mask_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, mask_dir)
	new_frame_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, frame_dir)
	create_dir(new_mask_path)
	create_dir(new_frame_path)

	# 2. flip mask and frame, and then copy
	for mask in mask_list:
		_, _mask = os.path.split(mask)
		mask_img = Image.open(mask)
		img_mirror = ImageOps.mirror(mask_img)
		img_mirror_path = os.path.join(new_mask_path, _mask)
		img_mirror.save(img_mirror_path, quality=95)

	for frame in frame_list:
		_, _frame = os.path.split(frame)
		frame_img = Image.open(frame)
		frame_mirror = ImageOps.mirror(frame_img)
		frame_mirror_path = os.path.join(new_frame_path, _frame)
		frame_mirror.save(frame_mirror_path, quality=95)
