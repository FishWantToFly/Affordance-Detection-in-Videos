'''
Do data augmentation on dataset_original 
Use 
    1. horizontally flip 
		Do on dataset_original only
		Need to copy : 
			mask -> flip
			inpaint_depth -> flip
			raw_frames -> flip
    2. change brightness 
    3. video clip of consecutive frame sequences
'''

import glob, os, copy, random
import numpy as np
from os import walk
from PIL import Image, ImageOps, ImageEnhance

def create_dir(new_dir):
	if not os.path.exists(new_dir):
		os.mkdir(new_dir)


# ######################################################
# # Step 1 : horizontally flip 
# test_action = "/home/s5078345/affordance/dataset/dataset_original/lab/chair_1/put_big_object_on_it"
# depth_dir = 'raw_depth'
# frame_dir = 'raw_frames'
# mask_dir = 'mask'
# depth_dir = 'inpaint_depth'
# now_dir = os.getcwd()
# copy_dir_name = 'dataset_horizontally_flip'
# if not os.path.exists(copy_dir_name):
# 	os.mkdir(copy_dir_name)

# for action in glob.glob("./dataset_original/*/*/*"):
# 	_, _action = os.path.split(action)
# 	print(action)

# 	# 1. create directory
# 	place = action.split('/')[-3]
# 	_object = action.split('/')[-2]
# 	# print(place)
# 	# print(_object)
# 	new_place_dir = os.path.join(now_dir, copy_dir_name, place)
# 	new_object_dir = os.path.join(now_dir, copy_dir_name, place, _object)
# 	if not os.path.exists(new_place_dir):
# 		os.mkdir(new_place_dir)
# 	if not os.path.exists(new_object_dir):
# 		os.mkdir(new_object_dir)

# 	action_mask_path = os.path.join(action, mask_dir, '*')
# 	action_frame_path = os.path.join(action, frame_dir, '*')
# 	action_depth_path = os.path.join(action, depth_dir, '*')
# 	mask_list = sorted(glob.glob(action_mask_path))
# 	frame_list = sorted(glob.glob(action_frame_path))
# 	depth_list = sorted(glob.glob(action_depth_path))

# 	new_action_path = os.path.join(now_dir, copy_dir_name, place, _object, _action)
# 	create_dir(new_action_path)
# 	new_mask_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, mask_dir)
# 	new_depth_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, depth_dir)
# 	new_frame_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, frame_dir)
# 	create_dir(new_mask_path)
# 	create_dir(new_depth_path)
# 	create_dir(new_frame_path)

# 	# 2. flip mask and frame, then copy
# 	for mask in mask_list:
# 		_, _mask = os.path.split(mask)
# 		mask_img = Image.open(mask)
# 		img_mirror = ImageOps.mirror(mask_img)
# 		img_mirror_path = os.path.join(new_mask_path, _mask)
# 		img_mirror.save(img_mirror_path, quality=95)

# 	for frame in frame_list:
# 		_, _frame = os.path.split(frame)
# 		frame_img = Image.open(frame)
# 		frame_mirror = ImageOps.mirror(frame_img)
# 		frame_mirror_path = os.path.join(new_frame_path, _frame)
# 		frame_mirror.save(frame_mirror_path, quality=95)

# 	# 3. flip depth then copy
# 	for depth in depth_list:
# 		_, _depth = os.path.split(depth)
# 		depth_npy = np.load(depth)
# 		depth_npy = depth_npy.reshape(480, 640)
# 		depth_mirror = np.fliplr(depth_npy)
# 		depth_mirror = depth_mirror.reshape(480, 640, 1)
# 		depth_mirror_path = os.path.join(new_depth_path, _depth)
# 		np.save(depth_mirror_path, depth_mirror)

######################################################
# Step 1 : horizontally flip 
test_action = "/home/s5078345/affordance/dataset/dataset_original/lab/chair_1/put_big_object_on_it"
depth_dir = 'raw_depth'
frame_dir = 'raw_frames'
mask_dir = 'mask'
depth_dir = 'inpaint_depth'
now_dir = os.getcwd()
copy_dir_name = 'dataset_flip_brightness_contrast'
if not os.path.exists(copy_dir_name):
	os.mkdir(copy_dir_name)

for action in glob.glob("./dataset_horizontally_flip/*/*/*"):
	# action = test_action
	_, _action = os.path.split(action)
	print(action)

	# 1. create directory
	place = action.split('/')[-3]
	_object = action.split('/')[-2]
	# print(place)
	# print(_object)
	new_place_dir = os.path.join(now_dir, copy_dir_name, place)
	new_object_dir = os.path.join(now_dir, copy_dir_name, place, _object)
	if not os.path.exists(new_place_dir):
		os.mkdir(new_place_dir)
	if not os.path.exists(new_object_dir):
		os.mkdir(new_object_dir)

	action_mask_path = os.path.join(action, mask_dir, '*')
	action_frame_path = os.path.join(action, frame_dir, '*')
	action_depth_path = os.path.join(action, depth_dir, '*')
	mask_list = sorted(glob.glob(action_mask_path))
	frame_list = sorted(glob.glob(action_frame_path))
	depth_list = sorted(glob.glob(action_depth_path))

	new_action_path = os.path.join(now_dir, copy_dir_name, place, _object, _action)
	create_dir(new_action_path)
	new_mask_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, mask_dir)
	new_depth_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, depth_dir)
	new_frame_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, frame_dir)
	create_dir(new_mask_path)
	create_dir(new_depth_path)
	create_dir(new_frame_path)

	random.seed()
	random_1 = random.random()
	random_2 = random.random()

	for frame in frame_list:
		_, _frame = os.path.split(frame)
		frame_img = Image.open(frame)

		# Adjust sharpness or brightnesss
		if random_1 > 0.6 :
			frame_img = ImageEnhance.Sharpness(frame_img)
			frame_img = frame_img.enhance(10.0)		
		else :
			frame_img = ImageEnhance.Brightness(frame_img)
			if random_2 > 0.5 :
				frame_img = frame_img.enhance(1.35)
			else :
				frame_img = frame_img.enhance(0.8)

		enhance_frame_path = os.path.join(new_frame_path, _frame)
		frame_img.save(enhance_frame_path, quality=95)

	# 2. just copy
	for mask in mask_list:
		_, _mask = os.path.split(mask)
		mask_img = Image.open(mask)
		img_mirror = mask_img
		img_mirror_path = os.path.join(new_mask_path, _mask)
		img_mirror.save(img_mirror_path, quality=95)

	# 3. just copy
	for depth in depth_list:
		_, _depth = os.path.split(depth)
		depth_npy = np.load(depth)
		depth_npy = depth_npy.reshape(480, 640)
		depth_mirror = depth_npy
		depth_mirror = depth_mirror.reshape(480, 640, 1)
		depth_mirror_path = os.path.join(new_depth_path, _depth)
		np.save(depth_mirror_path, depth_mirror)
		