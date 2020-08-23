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

print("For original dataset. (No count for augmented dataset.)\n")

######################################################
# Step 1 : visualize from raw depth information (transform to greyscale image)
test_action = "/home/s5078345/affordance/dataset/lab/chair_1/move_object_with_thing_in_it/"
depth_dir = 'raw_depth'
frame_dir = 'raw_frames'
mask_dir = 'mask'

action_total = 0
action_unique_list = []
action_frames_list = []
action_dict = {}
for action in glob.glob("../dataset_two_steps/dataset_original/*/*/*"):	


	# count unique action and record
	_, _action = os.path.split(action)
	x = re.search("_[0-9]+", _action)

	semantic = os.path.basename(os.path.dirname(action)).split('_')[0]
	if semantic == 'chair' or semantic == 'table' :
		pass
	else :
		continue
	# print(semantic)

	# count total training action
	action_total += 1

	if x is not None :
		_action = _action[:-2]
	if _action not in action_unique_list:
		action_unique_list.append(_action)

	if action_dict.get(_action) == None :
		action_dict[_action] = 1
	else :
		temp_num = action_dict[_action]
		action_dict[_action] = temp_num + 1

	# count frame number of each action
	action_frames = glob.glob(os.path.join(action, frame_dir, '*'))
	frames_number = len(action_frames)
	action_frames_list.append(frames_number)

	# if frames_number < 6:
	# 	print(action)

	# find cases that does not label mask
		# if not os.path.exists(os.path.join(action, mask_dir)):
		# 	print(action)


print("\nTotal action number = %d" % (action_total))
print("Total unique action number = %d" % (len(action_unique_list)))
print("Action list : ")
print(action_unique_list)

print("=== RGB Frame ===")
print("Total frame number = %d" % (sum(action_frames_list)))
print("Average frame number = %d" % (statistics.mean(action_frames_list)))
print("Maximum frame number = %d" % (max(action_frames_list)))
print("Minimum frame number = %d" % (min(action_frames_list)))

print("=== Action Dict ===")
print(action_dict)