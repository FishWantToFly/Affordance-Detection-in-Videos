'''
2020.3.10
Want to find out why val IoU can achieve high performance
Split train / test to mutually exclusive
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
datasets = ['dataset_original', 'dataset_horizontally_flip', 'dataset_flip_brightness_contrast', 'dataset_brightness_contrast']
train_places = ['home_living_room', 'kitchen', 'lab']
test_places = ['my_room']
for dataset in datasets:
	for train_place in train_places :
		for action in glob.glob("./%s/%s/*/*" % (dataset, train_place)):
			# action = test_action
			_, _action = os.path.split(action)
			# print(action)
			train_list.append(action)	
	for test_place in test_places :
		for action in glob.glob("./%s/%s/*/*" % (dataset, test_place)):
			# action = test_action
			_, _action = os.path.split(action)
			# print(action)
			test_list.append(action)

# random.seed()
# train_list, test_list = train_test_split(data_list, test_size=0.2)
print("Train data len : %d" % (len(train_list)))
print("Test data len : %d" % (len(test_list)))

train_save_dir = os.path.join(data_list_dir, 'train_list_v2.txt')
test_save_dir = os.path.join(data_list_dir, 'test_list_v2.txt')
with open(train_save_dir, 'w') as f:
	for action in train_list:
		f.write("%s\n" % action)
with open(test_save_dir, 'w') as f:
	for action in test_list:
		f.write("%s\n" % action)
