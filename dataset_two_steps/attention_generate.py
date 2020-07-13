'''
2020.7.13

generate affordance attention
對於之前會改變大小的action，一律使用最大範圍的mask
因為支撐的物體位置不會變，直接複製開頭 or 結尾的mask就好了
e.g. 
    複製開頭：
        Put Object on It
        Put Big Object on It
        People Sit in It
        Put and Remove Object on It

    複製結尾：
        Remove Object on It
        Remove Big Object on It
        People Stand Up From It
        Remove and Put Object on It


'''

import glob, os
import numpy as np
from os import walk
from PIL import Image, ImageOps, ImageEnhance

def create_dir(new_dir):
	if not os.path.exists(new_dir):
		os.mkdir(new_dir)

old_mask_name = 'mask'
attention_mask_name = 'attention_mask'
# mask_rgb_name = 'mask_rgb'
# first_mask_name = 'first_mask'
# first_mask_rgb_name = 'first_mask_rgb'

print("Step 1")
actions = sorted(glob.glob("./dataset_original/*/*/*"))
for action in actions:
    _object = action.split('/')[3].split('_')[0]
    target_list = ['chair', 'table']
    if _object not in target_list:
        continue
        
    # source_action_path = os.path.join(dataset_source, ('/').join(action.split('/')[2:5]))
    
    # for test
    action = './dataset_original/home_living_room/chair_1/move_object_itself'
    print(action)

    old_mask_path = os.path.join(action, old_mask_name) # old
    attention_mask_path = os.path.join(action, attention_mask_name)
    create_dir(attention_mask_path)




    # # 1. chehck whether there are first_mask and first_mask_rgb
    # if not os.path.exists(os.path.join(action, first_mask_name)) and not os.path.exists(os.path.join(action, first_mask_rgb_name)) :
    #     # directly copy from dataset_source
    #     os.system("cp -r %s %s" % (mask_path, first_mask_path))
    #     os.system("cp -r %s %s" % (mask_rgb_path, first_mask_rgb_path))
    # else :
    #     # 2. check if there is missing first_mask_rgb. If miss, copy from mask_rgb
    #     # 3. check if there is missing first_mask. If miss, copy from mask

    #     # List raw_frames and map to mask and mask_rgb
    #     frames = sorted(glob.glob(os.path.join(action, 'raw_frames/*')))
    #     for frame in frames : 
    #         frame_name = os.path.split(frame)[1]
    #         first_mask_now = os.path.join(action, first_mask_name, frame_name[:-4] + '.jpg')
    #         first_mask_rgb_now = os.path.join(action, first_mask_rgb_name, frame_name[:-4] + '.jpg')
    #         mask_now = os.path.join(action, mask_name, frame_name[:-4] + '.jpg')
    #         mask_rgb_now = os.path.join(action, mask_rgb_name, frame_name[:-4] + '.jpg')

    #         # chech first_mask_rgb
    #         if not os.path.exists(first_mask_rgb_now):
    #             os.system("cp -r %s %s" % (mask_rgb_now, first_mask_rgb_now))
    #         # chech first_mask
    #         if not os.path.exists(first_mask_now):
    #             os.system("cp -r %s %s" % (mask_now, first_mask_now))

    break

