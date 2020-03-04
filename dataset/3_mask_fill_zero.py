'''
fill all black mask to 
1. no corresponding frame -> mask
2. no existence of directory of mask
'''

import cv2, glob, os, copy
import numpy as np
from os import walk
from PIL import Image

######################################################
# Step 1 : visualize from raw depth information (transform to greyscale image)
test_action = "/home/s5078345/affordance/dataset/lab/chair_1/move_object_with_thing_in_it/"
depth_dir = 'raw_depth'
frame_dir = 'raw_frames'
mask_dir = 'mask'

black_image_path = "/home/s5078345/affordance/dataset/all_black_image.jpg"
black_image = Image.open(black_image_path)
black_image = black_image.convert("RGB")

# delete redundant depth
for action in glob.glob("./*/*/*"):
	
	print(action)
	action_mask_path = os.path.join(action, mask_dir, '*')
	action_frame_path = os.path.join(action, frame_dir, '*')
	# mask_list = sorted(glob.glob(action_mask_path))
	frame_list = sorted(glob.glob(action_frame_path))

	# # 1. no mask for this action
	if not os.path.exists(os.path.join(action, mask_dir)):
		os.mkdir(os.path.join(action, mask_dir))
		for frame in frame_list :
			frame_name = frame.split('/')[-1]
			maybe_exist_mask = os.path.join(action, mask_dir, frame_name[:-4] + '.jpg')
			black_image.save(maybe_exist_mask)

	else :
	# 2. some mask are not generated
		for frame in frame_list :
			frame_name = frame.split('/')[-1]
			maybe_exist_mask = os.path.join(action, mask_dir, frame_name[:-4] + '.jpg')
			if not os.path.isfile(maybe_exist_mask):
				black_image.save(maybe_exist_mask)
