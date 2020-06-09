'''
2020.6.6

original : 100 and 109

v1 : all about 8000 images
'''

import glob, os, copy, random
import numpy as np
from os import walk
from sklearn.model_selection import train_test_split
from PIL import Image

def create_dir(new_dir):
	if not os.path.exists(new_dir):
		os.mkdir(new_dir)

data_list_dir = "./data_list"
create_dir(data_list_dir)

depth_dir = 'raw_depth'
now_dir = os.getcwd()

data_list, train_list, test_list = [], [], []

'''
v4
'''
datasets = ['dataset_original']
# dataset_horizontally_flip

# place = 'coco'
# train_save_name = 'train_list.txt'
# train_10_save_name = 'train_list_10.txt'
# test_save_name = 'test_list.txt'
# test_10_save_name = 'test_list_10.txt'

place = 'coco_all'
train_save_name = 'train_list_v1.txt'
train_10_save_name = 'train_list_10_v1.txt'
test_save_name = 'test_list_v1.txt'
test_10_save_name = 'test_list_10_v1.txt'

temp = 0
for dataset in datasets:
	for action in glob.glob("./%s/%s/*/*" % (dataset, place)):
		place = action.split("/")[2]
		_object_semantic = action.split("/")[3].split('_')[0]
		_object = action.split("/")[3]

		# check weather it is rgb image
		img = np.array(Image.open(os.path.join(action, 'raw_frames/00000.png')))
		if img.ndim != 3 : continue
		# print(img.ndim)

		data_list.append(action)
		temp += 1
		# if temp > 5001 : break

# pick 5000 right now (train : 4000 / test : 1000)
# data_list = data_list[:5000]

_train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=0)

# add flip images into training data list
for action in _train_list :
	flip_action = os.path.join('dataset_horizontally_flip', action.split('/')[2], action.split('/')[3], action.split('/')[4])
	train_list.append(action)
	train_list.append(flip_action)

print("Train data len : %d" % (len(train_list)))
print("Test data len : %d" % (len(test_list)))

train_save_dir = os.path.join(data_list_dir, train_save_name)
test_save_dir = os.path.join(data_list_dir, test_save_name)
with open(train_save_dir, 'w') as f:
	for action in train_list:
		f.write("%s\n" % action)
with open(test_save_dir, 'w') as f:
	for action in test_list:
		f.write("%s\n" % action)
train_10_save_dir = os.path.join(data_list_dir, train_10_save_name)
test_10_save_dir = os.path.join(data_list_dir, test_10_save_name)
with open(train_10_save_dir, 'w') as f:
	for action in train_list[:10]:
		f.write("%s\n" % action)
with open(test_10_save_dir, 'w') as f:
	for action in test_list[:10]:
		f.write("%s\n" % action)

