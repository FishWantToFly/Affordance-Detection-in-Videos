'''
2020.3.10
Want to find out why val IoU can achieve high performance
Split train / test to mutually exclusive

2020.3.23
1. just use chair and table as training / testing data
2. use specific place and object as testing data, other as training data

2020.3.26
For test data, we just test on original one

2020.4.11
just list original data (not augmented)
'''

import glob, os, copy, random
import numpy as np
from os import walk
from sklearn.model_selection import train_test_split

def create_dir(new_dir):
	if not os.path.exists(new_dir):
		os.mkdir(new_dir)

data_list_dir = "./data_list"
create_dir(data_list_dir)

depth_dir = 'raw_depth'
frame_dir = 'raw_frames'
mask_dir = 'mask'
depth_dir = 'inpaint_depth'
now_dir = os.getcwd()

data_list, train_list, test_list = [], [], []
# '''
# v3
# '''
# datasets = ['dataset_original', 'dataset_horizontally_flip', 'dataset_flip_brightness_contrast', 'dataset_brightness_contrast', \
# 'dataset_original_so', 'dataset_horizontally_flip_so', 'dataset_flip_brightness_contrast_so', 'dataset_brightness_contrast_so']

# '''
# v4
# '''
# datasets = ['dataset_original', 'dataset_horizontally_flip', 'dataset_flip_brightness_contrast', 'dataset_brightness_contrast']

'''
v4
'''
datasets = ['dataset_original', 'dataset_horizontally_flip']

train_test_objects = ['table', 'chair']
test_place_object_list = []
# for testing
test_place_object_list.append(['tohoku_lab', 'table'])
test_place_object_list.append(['lab', 'chair'] )

# test new add
test_place_object_list.append(['tohoku_meeting_room', 'table'])
test_place_object_list.append(['my_room', 'chair'] )

for dataset in datasets:
	for action in glob.glob("./%s/*/*/*" % (dataset)):
		place = action.split("/")[2]
		_object = action.split("/")[3].split('_')[0]

		## only pick chair or table
		if _object not in train_test_objects:
			continue

		## choose train or test
		if [place, _object] in test_place_object_list and dataset == 'dataset_original': 
			test_list.append(action)
		elif [place, _object] not in test_place_object_list:
			train_list.append(action)

print("Train data len : %d" % (len(train_list)))
print("Test data len : %d" % (len(test_list)))

train_save_dir = os.path.join(data_list_dir, 'train_list_v5.txt')
test_save_dir = os.path.join(data_list_dir, 'test_list_v5.txt')
with open(train_save_dir, 'w') as f:
	for action in train_list:
		f.write("%s\n" % action)
with open(test_save_dir, 'w') as f:
	for action in test_list:
		f.write("%s\n" % action)
train_10_save_dir = os.path.join(data_list_dir, 'train_list_10_v5.txt')
test_10_save_dir = os.path.join(data_list_dir, 'test_list_10_v5.txt')
with open(train_10_save_dir, 'w') as f:
	for action in train_list[:10]:
		f.write("%s\n" % action)
with open(test_10_save_dir, 'w') as f:
	for action in test_list[:10]:
		f.write("%s\n" % action)


## list original all data for table and chair
original_dataset_name = 'dataset_original'
data_list = []
for action in glob.glob("./%s/*/*/*" % (original_dataset_name)):
	_object = action.split("/")[3].split('_')[0]

	## only pick chair or table
	if _object not in train_test_objects:
		continue
	data_list.append(action)

all_save_dir = os.path.join(data_list_dir, 'original_data_list_v5.txt')
with open(all_save_dir, 'w') as f:
	for action in data_list:
		f.write("%s\n" % action)

print("Original data len = %d" % (len(data_list)))

'''
v2 
'''
# train_places = ['home_living_room', 'kitchen', 'lab']
# test_places = ['my_room']
# for dataset in datasets:
# 	for train_place in train_places :
# 		for action in glob.glob("./%s/%s/*/*" % (dataset, train_place)):
# 			# action = test_action
# 			_, _action = os.path.split(action)
# 			# print(action)
# 			train_list.append(action)	
# 	for test_place in test_places :
# 		for action in glob.glob("./%s/%s/*/*" % (dataset, test_place)):
# 			# action = test_action
# 			_, _action = os.path.split(action)
# 			# print(action)
# 			test_list.append(action)

# # random.seed()
# # train_list, test_list = train_test_split(data_list, test_size=0.2)
# print("Train data len : %d" % (len(train_list)))
# print("Test data len : %d" % (len(test_list)))

# train_save_dir = os.path.join(data_list_dir, 'train_list_v2.txt')
# test_save_dir = os.path.join(data_list_dir, 'test_list_v2.txt')
# with open(train_save_dir, 'w') as f:
# 	for action in train_list:
# 		f.write("%s\n" % action)
# with open(test_save_dir, 'w') as f:
# 	for action in test_list:
# 		f.write("%s\n" % action)
