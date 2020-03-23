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
# test_action = "/home/s5078345/affordance/dataset/lab/chair_1/move_object_with_thing_in_it/"
depth_dir = 'raw_depth'
frame_dir = 'raw_frames'
mask_dir = 'mask'

place_list = ['home_living_room', 'kitchen', 'lab', 'my_room', 'tohoku_lab', \
	'tohoku_meeting_room', 'tohoku_seminar_room']
for place in place_list :
	semantic_dict = {'basket': 0, 'chair': 0, 'plate': 0,'sofa': 0, 'table': 0}
	for action in glob.glob("../dataset/dataset_original/%s/*/*" % (place)):
		semantic = os.path.basename(os.path.dirname(action)).split('_')[0]
		semantic_dict[semantic] += 1
	print("Place : %s" % (place))
	print(semantic_dict)	
	print()