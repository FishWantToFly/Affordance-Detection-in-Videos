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

action_total = 0
action_unique_list = []
action_frames_list = []
for action in glob.glob("../dataset/dataset_original/*/*/*"):	
	# count total training action
	action_total += 1

	# count unique action and record
	_, _action = os.path.split(action)
	x = re.search("_[0-9]+", _action)
	if x is None and _action not in action_unique_list:
		action_unique_list.append(_action)

	# count frame number of each action
	action_frames = glob.glob(os.path.join(action, frame_dir, '*'))
	frames_number = len(action_frames)
	action_frames_list.append(frames_number)


print("Total action number = %d" % (action_total))
print("Total unique action number = %d" % (len(action_unique_list)))
print("Action list : ")
print(action_unique_list)

print("=== RGB Frame ===")
print("Total frame number = %d" % (sum(action_frames_list)))
print("Average frame number = %d" % (statistics.mean(action_frames_list)))
print("Maximum frame number = %d" % (max(action_frames_list)))
print("Minimum frame number = %d" % (min(action_frames_list)))