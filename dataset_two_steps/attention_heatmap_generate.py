'''
2020.7.13

after attention_generate.py

1. change attention into gaussian attention heatmap
2. flip attention heatmap into dataset_horizontally_flip folder

'''

import glob, os, re
import numpy as np
from os import walk
from PIL import Image, ImageOps, ImageEnhance
import cv2

def create_dir(new_dir):
	if not os.path.exists(new_dir):
		os.mkdir(new_dir)

old_mask_name = 'mask'
attention_mask_name = 'attention_mask'
attention_heatmap_name = 'attention_heatmap'


# 1. change attention into gaussian attention heatmap
'''
actions = sorted(glob.glob("./dataset_original/*/*/*"))
for action in actions:
    _object = action.split('/')[3].split('_')[0]
    target_list = ['chair', 'table']
    if _object not in target_list:
        continue
        
    # for test
    # action = './dataset_original/home_living_room/chair_1/move_object_itself'


    print(action)

    old_mask_path = os.path.join(action, old_mask_name) # old
    attention_mask_dir = os.path.join(action, attention_mask_name)
    attention_heatmap_dir = os.path.join(action, attention_heatmap_name)
    create_dir(attention_heatmap_dir)
    attention_masks = sorted(glob.glob(os.path.join('%s/*' % (attention_mask_dir))))

    for attention_mask in attention_masks :
        _, mask_pure_name = os.path.split(attention_mask)

        attention_heatmap_path = os.path.join(attention_heatmap_dir, mask_pure_name)

        img = cv2.imread(attention_mask)
        blur_img = cv2.GaussianBlur(img,(41,41), 0)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY) # from RGB to greyscale
        cv2.imwrite(attention_heatmap_path, blur_img)

        # heatmap = cv2.applyColorMap(blur_img, cv2.COLORMAP_HOT)
        # cv2.imwrite(os.path.join(attention_heatmap_dir, 'h_' + mask_pure_name), heatmap)
'''


# '''
# 2. flip attention heatmap into dataset_horizontally_flip folder
now_dir = os.getcwd()
copy_dir_name = 'dataset_horizontally_flip'
heatmap_dir = 'attention_heatmap'

actions = sorted(glob.glob("./dataset_original/*/*/*"))
for action in actions:
    _, _action = os.path.split(action)
    object_semantic = action.split('/')[3].split('_')[0]
    target_list = ['chair', 'table']
    if object_semantic not in target_list:
        continue
        
    # for test
    # action = './dataset_original/home_living_room/chair_1/move_object_itself'
    print(action)

    place = action.split('/')[-3]
    _object = action.split('/')[-2]

    # attention_heatmap_dir = os.path.join(action, attention_heatmap_name)
    # attention_heatmaps = sorted(glob.glob(os.path.join('%s/*' % (attention_heatmap_dir))))
    # new_heatmap_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, heatmap_dir)
    # create_dir(new_heatmap_path)

    # for attention_heatmap in attention_heatmaps :
    #     _, _heatmap = os.path.split(attention_heatmap)
    #     heaymap_img = Image.open(attention_heatmap)
    #     heatmap_mirror = ImageOps.mirror(heaymap_img)
    #     heatmap_mirror_path = os.path.join(new_heatmap_path, _heatmap)
    #     heatmap_mirror.save(heatmap_mirror_path, quality=95)

    # copy attention mask
    attention_mask_dir = os.path.join(action, attention_mask_name)
    attention_masks = sorted(glob.glob(os.path.join('%s/*' % (attention_mask_dir))))
    new_mask_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, attention_mask_name)
    create_dir(new_mask_path)

    for attention_mask in attention_masks :
        _, _mask = os.path.split(attention_mask)
        mask_img = Image.open(attention_mask)
        mask_mirror = ImageOps.mirror(mask_img)
        mask_mirror_path = os.path.join(new_mask_path, _mask)
        mask_mirror.save(mask_mirror_path, quality=95)

    # break
# '''
