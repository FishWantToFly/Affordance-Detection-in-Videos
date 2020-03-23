'''
have to open a nwe terminal and ...
source activate siammask

inpaint depth
	2.1 transform to greyscale image (.txt -> .npy -> .png)
	2.2 use cv2.inpaint to inpaint greyscale image
	2.3 visulaize to check (.png)
	2.4 transform back to actual depth info. (.npy)
'''

## step 0 : have to use siammask environment to use cv2
import cv2, glob, os, copy
import numpy as np
from os import walk
from PIL import Image

# dataset_name = 'dataset_lab_ito'
dataset_name = 'tohoku'

######################################################
# Step 1 : visualize from raw depth information (transform to greyscale image)
# test_action = "/home/s5078345/affordance/dataset/kitchen/chair_1/remove_big_object_on_it_1/"
depth_dir = 'raw_depth'
frame_dir = 'raw_frames'

for action in glob.glob("./%s/*/*/*" % (dataset_name)):
	print(action)
	action_depth_path = os.path.join(action, depth_dir, '*')
	action_frame_path = os.path.join(action, frame_dir, '*')
	depth_list = sorted(glob.glob(action_depth_path))
	frame_list = sorted(glob.glob(action_frame_path))

	for depth_txt in depth_list :
		depth_txt_name = depth_txt.split('/')[-1] 
		depth_list = []
		# print(depth_txt_name)
		with open(depth_txt) as f:
			for line in f:
				inner_list = [elt.strip() for elt in line.split(',')]
				depth_list.append(inner_list)
		depth_array = np.array(depth_list, dtype = np.int)
		depth_max_value = np.amax(depth_array)
		depth_array = np.reshape(depth_array, (480, 640))
		
		# create inpaint mask
		mask = copy.deepcopy(depth_array)
		for i in range (mask.shape[0]): # traverses through height of the image
			for j in range (mask.shape[1]): # traverses through width of the image
				if mask[i][j] != 0:
					mask[i][j] = 0
				else :
					mask[i][j] = 255
		mask = mask.astype(np.uint8)

		# save as greyscale image
		depth_array_greyscale = depth_array * 255 / depth_max_value
		depth_array_greyscale = np.asarray(depth_array_greyscale, dtype = np.int)

		# inpaint
		depth_array_greyscale = np.reshape(depth_array_greyscale, (480, 640, 1)).astype(np.uint8)
		inpaint_depth = cv2.inpaint(depth_array_greyscale, mask, 3, cv2.INPAINT_TELEA)

		# visualization inpaint result
		inpaint_depth_png_folder = os.path.join(action, 'inpaint_depth_png')
		if not os.path.exists(inpaint_depth_png_folder):
			os.mkdir(inpaint_depth_png_folder)
		cv2.imwrite(os.path.join(inpaint_depth_png_folder, depth_txt_name[:-4] + '.png'), inpaint_depth)

		# rescale back to original scale
		depth_array_npy = np.asarray(depth_array_greyscale, dtype = np.int)
		depth_array_npy = (depth_array_npy * depth_max_value / 255).astype(np.int)
		depth_array_npy_dir = os.path.join(action, 'inpaint_depth')
		if not os.path.exists(depth_array_npy_dir):
			os.mkdir(depth_array_npy_dir)
		np.save(os.path.join(depth_array_npy_dir, depth_txt_name[:-4] + '.npy'), depth_array_npy) # 480 x 640 x 1
