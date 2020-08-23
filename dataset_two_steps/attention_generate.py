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

    其他的直接複製就好
        下列的要先查看attention_mask裡面有沒有相對應的mask
            Move Object With Thing In It
            Pick up It
            Push down It

'''

import glob, os, re
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
    # action = './dataset_original/home_living_room/chair_1/move_object_itself'
    # action = './dataset_original/home_living_room/chair_1/put_object_on_it_1'
    # action = './dataset_original/home_living_room/chair_1/pick_up_it'
    # action = './dataset_original/tohoku_lab/chair_1/remove_and_put_object_on_it_1'

    print(action)
    _, _action = os.path.split(action)
    x = re.search("_[0-9]+", _action)
    if x is not None :
        _action = _action[:-2] # pure action name

    old_mask_path = os.path.join(action, old_mask_name) # old
    attention_mask_dir = os.path.join(action, attention_mask_name)
    create_dir(attention_mask_dir)
    old_masks = sorted(glob.glob(os.path.join(action, '%s/*' % (old_mask_name))))
    
    if _action in ['put_object_on_it', 'remove_object_on_it', 'people_stand_up_from_it', 'people_sit_on_it', 'put_and_remove_object_on_it', \
        'remove_and_put_object_on_it', 'put_big_object_on_it', 'remove_big_object_on_it'] :
        # copy first or mask mask
        first_mask_path = old_masks[0]
        last_mask_path = old_masks[-1]
        for old_mask in old_masks :
            _, mask_pure_name = os.path.split(old_mask)
            attention_mask_path = os.path.join(attention_mask_dir, mask_pure_name)
            if _action in ['put_object_on_it', 'people_sit_on_it', 'put_and_remove_object_on_it', 'put_big_object_on_it']:
                # copy first mask
                copied_mask_path = first_mask_path
                os.system("cp -r %s %s" % (copied_mask_path, attention_mask_path))
            else :
                ## carefully deal with remove_and_put_object_on_it
                if action == './dataset_original/tohoku_lab/chair_1/remove_and_put_object_on_it_1' :
                    copied_mask_path = os.path.join(old_mask_path, '00006.jpg')
                elif action == './dataset_original/tohoku_lab/chair_1/remove_and_put_object_on_it_2' :
                    copied_mask_path = os.path.join(old_mask_path, '00013.jpg')

                elif action == './dataset_original/tohoku_lab/chair_1_angle_2/remove_and_put_object_on_it_1' :
                    copied_mask_path = os.path.join(old_mask_path, '00015.jpg')
                elif action == './dataset_original/tohoku_lab/chair_1_angle_2/remove_and_put_object_on_it_2' :
                    copied_mask_path = os.path.join(old_mask_path, '00010.jpg')
                elif action == './dataset_original/tohoku_lab/chair_1_angle_2/remove_and_put_object_on_it_3' :
                    copied_mask_path = os.path.join(old_mask_path, '00015.jpg')

                elif action == './dataset_original/tohoku_lab/chair_1_angle_3/remove_and_put_object_on_it_1' :
                    copied_mask_path = os.path.join(old_mask_path, '00015.jpg')

                elif action == './dataset_original/tohoku_lab/chair_2/remove_and_put_object_on_it_1' :
                    copied_mask_path = os.path.join(old_mask_path, '00012.jpg')

                elif action == './dataset_original/tohoku_lab/table/remove_and_put_object_on_it_1' :
                    copied_mask_path = os.path.join(old_mask_path, '00007.jpg')

                elif action == './dataset_original/tohoku_meeting_room/table_1/remove_and_put_object_on_it_1' :
                    copied_mask_path = os.path.join(old_mask_path, '00014.jpg')

                elif action == './dataset_original/tohoku_seminar_room/chair_1/remove_and_put_object_on_it_1' :
                    copied_mask_path = os.path.join(old_mask_path, '00005.jpg')
                elif action == './dataset_original/tohoku_seminar_room/chair_2/remove_and_put_object_on_it_1' :
                    copied_mask_path = os.path.join(old_mask_path, '00006.jpg')
                elif action == './dataset_original/tohoku_seminar_room/table_2_angle_2/remove_and_put_object_on_it_1' :
                    copied_mask_path = os.path.join(old_mask_path, '00014.jpg')

                else :
                # copy last mask
                    copied_mask_path = last_mask_path
                os.system("cp -r %s %s" % (copied_mask_path, attention_mask_path))


            # print(copied_mask_path)
            # print(attention_mask_path)
            # print()

    else :
        for old_mask in old_masks :
            _, mask_pure_name = os.path.split(old_mask)
            attention_mask_path = os.path.join(attention_mask_dir, mask_pure_name)
            if _action in ['move_object_with_thing_in_it', 'pick_up_it', 'push_down_it'] :
                # firstly check is there any esisted mask in attention_mask dir
                if os.path.exists(attention_mask_path):
                    # print("> <")
                    continue 
            copied_mask_path = old_mask
            os.system("cp -r %s %s" % (copied_mask_path, attention_mask_path))
    
            # print(copied_mask_path)
            # print(attention_mask_path)
            # print()

    # break

