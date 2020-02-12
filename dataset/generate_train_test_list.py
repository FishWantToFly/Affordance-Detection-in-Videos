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

data_list = []
for action in glob.glob("./*/*/*/*"):
	# action = test_action
	_, _action = os.path.split(action)
	# print(action)
	data_list.append(action)

random.seed()
train_list, test_list = train_test_split(data_list, test_size=0.2)
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