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
test_action = "/home/s5078345/Affordance-Detection-on-Video/dataset/dataset_original/lab/chair_1/move_object_with_thing_in_it/"
depth_dir = 'inpaint_depth'
frame_dir = 'raw_frames'
mask_dir = 'mask'

depth_max = -1
for action in glob.glob("../dataset/dataset_original/*/*/*"):	

	# count unique action and record
	_, _action = os.path.split(action)


	# count frame number of each action
	action_depths = glob.glob(os.path.join(action, depth_dir, '*'))
	for depth_file in action_depths:
		depth = np.load(depth_file)
		temp_depth_max = np.max(depth)
		if temp_depth_max > depth_max:
			depth_max = temp_depth_max
			print(depth_max)

print("==========================")
print(depth_max) # 9897


