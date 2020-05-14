'''
2020.4.28
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
depth_dir = 'inpaint_depth'
now_dir = os.getcwd()

data_list, train_list, test_list = [], [], []

'''
v4
'''
datasets = ['dataset_original', 'dataset_horizontally_flip']

train_test_objects = ['table', 'chair']
test_place_object_list = []

# for testing
# test_place_object_list.append(['tohoku_lab', 'table'])
# test_place_object_list.append(['lab', 'chair'] )
# test_place_object_list.append(['tohoku_meeting_room', 'table'])
# test_place_object_list.append(['my_room', 'chair'] )

# new
test_place_object_list.append(['tohoku_seminar_room', 'table_2'])
test_place_object_list.append(['tohoku_seminar_room', 'table_2_angle_2'])
test_place_object_list.append(['my_room', 'table_1'])

test_place_object_list.append(['lab', 'chair_1'])
test_place_object_list.append(['lab', 'chair_2'])
test_place_object_list.append(['home_living_room', 'chair_1'])
test_place_object_list.append(['tohoku_seminar_room', 'chair_1'])

for dataset in datasets:
	for action in glob.glob("./%s/*/*/*" % (dataset)):
		place = action.split("/")[2]
		_object_semantic = action.split("/")[3].split('_')[0]
		_object = action.split("/")[3]
		print(_object)

		## only pick chair or table
		if _object_semantic not in train_test_objects:
			continue

		## choose train or test
		if [place, _object] in test_place_object_list and dataset == 'dataset_original': 
			test_list.append(action)
		elif [place, _object] not in test_place_object_list:
			train_list.append(action)

print("Train data len : %d" % (len(train_list)))
print("Test data len : %d" % (len(test_list)))

train_save_dir = os.path.join(data_list_dir, 'train_list.txt')
test_save_dir = os.path.join(data_list_dir, 'test_list.txt')
with open(train_save_dir, 'w') as f:
	for action in train_list:
		f.write("%s\n" % action)
with open(test_save_dir, 'w') as f:
	for action in test_list:
		f.write("%s\n" % action)
train_10_save_dir = os.path.join(data_list_dir, 'train_list_10.txt')
test_10_save_dir = os.path.join(data_list_dir, 'test_list_10.txt')
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

all_save_dir = os.path.join(data_list_dir, 'original_data_list.txt')
with open(all_save_dir, 'w') as f:
	for action in data_list:
		f.write("%s\n" % action)

print("Original data len = %d" % (len(data_list)))
