'''
2020.5.23

Evaluate for step 1 + step 2 
Step 1 output is pre-generated (from args.mask)

'''

from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math
import glob

import torch
import torch.utils.data as data

from affordance.utils.osutils import *
from affordance.utils.imutils import *
from affordance.utils.transforms import *


class Sad_eval(data.Dataset):
    def __init__(self, is_train = True, **kwargs):
        self.img_folder = kwargs['image_path'] # root image folders
        self.is_train   = is_train # training set or test set
        self.inp_res    = kwargs['inp_res']
        self.out_res    = kwargs['out_res']
        self.sigma      = kwargs['sigma']
        self.dataset_list_dir_path = kwargs['dataset_list_dir_path']
        self.input_mask_dir = kwargs['mask']

        self.semantic_dict = {'basket': 0, 'chair': 1, 'plate': 2, 'sofa': 3, 'table': 4}
        self.semantic_len = len(self.semantic_dict)

        # contain img and annotation

        if kwargs['relabel']: # for relabel / visualization
            self.train_list = self.load_full_file_list('test_list') # dummy
            self.valid_list = self.load_full_file_list('test_list')

        elif kwargs['test'] == True:
            self.train_list = self.load_full_file_list('train_list_10')
            self.valid_list = self.load_full_file_list('test_list_10')
        else :
            self.train_list = self.load_full_file_list('train_list')
            self.valid_list = self.load_full_file_list('test_list')
            print("Train set number : %d" % len(self.train_list))
            print("Test set number : %d" % len(self.valid_list))


    def load_full_file_list(self, data_list):
        all_files = []
        action_list = []
        read_action_list_dir = os.path.join(self.dataset_list_dir_path, data_list + '.txt')
        with open(read_action_list_dir) as f:
            for line in f:
                inner_list = [elt.strip() for elt in line.split(' ')]
                action_list.append(inner_list[0])

        image_dir_name = 'raw_frames'
        check_mask_dir_name = 'mask'
        mask_dir_name = 'first_mask' 
        depth_dir_name = 'inpaint_depth'
        mask_rgb_dir_name = 'mask_rgb'
        gt_mask_dir_name = 'mask' # for computing first step iou > 50%
        for action in action_list :
            action_rgb_frames = glob.glob(os.path.join(self.img_folder, action, image_dir_name, '*.png'))
            semantic_name = os.path.basename(os.path.dirname(action)).split('_')[0]
            semantic_label = self.semantic_dict[semantic_name]

            # print(action.split('/')[2:])
            input_action_path = ('/').join(action.split('/')[2:])

            # for video training
            # each time there are 6 feames, overlap = 2 frames
            sorted_action_rgb_frames = sorted(action_rgb_frames)
            start_frame = -4 # -> 0
            
            while (True) :
                start_frame += 4
                if (start_frame + 6) >= len(action_rgb_frames):
                    start_frame = (len(action_rgb_frames)) - 6

                temp = [] 
                for i in range(6):
                    _index = start_frame + i
                    frame = sorted_action_rgb_frames[start_frame + i] 
                    frame_name = os.path.basename(frame)
                    frame_dir_dir = os.path.dirname(os.path.dirname(frame))
                    # here can use gt mask of predcited mask
                    check_mask = os.path.join(frame_dir_dir, check_mask_dir_name, frame_name[:-4] + '.jpg')
                    depth = os.path.join(frame_dir_dir, depth_dir_name, frame_name[:-4] + '.npy')
                    gt_mask = os.path.join(frame_dir_dir, gt_mask_dir_name, frame_name[:-4] + '.jpg')
                
                    input_mask = os.path.join(self.input_mask_dir, input_action_path, frame_name[:-4] + '.jpg')
                    
                    # frame_dir_dir : /home/s5078345/Affordance-Detection-on-Video/dataset_two_steps/./dataset_original/home_living_room/chair_2_angle_2/remove_object_on_it_1
                    # NEED TO FIND CLASSIFICATION GT FROM dataset_original mask and mask_rgb
                    # if mask_rgb None : false. if mask_rgb exists image : true
                    origin_dataset_name = 'dataset_original'
                    temp_1 = ('/').join(frame_dir_dir.split('/')[0:6])
                    temp_2 = ('/').join(frame_dir_dir.split('/')[7:])
                    origin_frame_dir_dir = os.path.join(temp_1, origin_dataset_name, temp_2)
                    mask_rgb = os.path.join(frame_dir_dir, mask_rgb_dir_name, frame_name[:-4] + '.jpg')

                    affordance_label = None
                    if os.path.exists(check_mask) and os.path.exists(mask_rgb) :
                        affordance_label = True
                    elif os.path.exists(check_mask) and not os.path.exists(mask_rgb) :
                        affordance_label = False 
                    else :
                        print("Mask got wrong QQ")

                    # temp.append([frame, gt_mask, depth, affordance_label, _index])　# use gt mask as input
                    temp.append([frame, input_mask, depth, affordance_label, _index, gt_mask]) # use pred mask as input
                    
                all_files.append(temp)

                # reach the end
                if (start_frame + 6) == len(action_rgb_frames):
                    break
        return all_files
    
    def __getitem__(self, index):
        video_len = 6
        mask_path_list = []
        image_index_list = []
        gt_mask_path_list = []

        # img_path, mask_path, depth_path = self.train_list[index]
        if self.is_train:
            video_data = self.train_list[index]
        else :
            video_data = self.valid_list[index]

        video_input = torch.zeros(video_len, 3, self.inp_res, self.inp_res)
        video_input_depth = torch.zeros(video_len, 1, self.inp_res, self.inp_res)
        video_input_mask = torch.zeros(video_len, 1, self.inp_res, self.inp_res)
        video_gt_mask = torch.zeros(video_len, 1, self.inp_res, self.inp_res)
        video_target_label = torch.zeros(video_len, 1)

        for i in range(video_len):
            img_path, mask_path, depth_path, affordance_label, _index, gt_mask_path = video_data[i]

            # load image and mask
            img = load_image(img_path)  # CxHxW
            a = load_mask(mask_path)    # 1xHxW
            depth = load_depth(depth_path)
            a_gt = load_mask(gt_mask_path)
            
            nparts = 1 # should change if target is more than 2

            # Prepare image and groundtruth map
            # inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
            inp = resize(img, self.inp_res, self.inp_res) # get normalized rgb value
            input_depth = resize(depth, self.inp_res, self.inp_res)
            # Generate input mask
            input_mask = torch.zeros(nparts, self.inp_res, self.inp_res) # [1, out_res, out_res]
            _input_mask = a[0] # HxW
            input_mask[0]  = resize(_input_mask, self.inp_res, self.inp_res) # [inp_res, inp_res]

            gt_mask = torch.zeros(nparts, self.inp_res, self.inp_res) # [1, out_res, out_res]
            _gt_mask = a_gt[0] # HxW
            gt_mask[0]  = resize(_gt_mask, self.inp_res, self.inp_res) # [inp_res, inp_res]            

            ##############################################
            # Output
            video_input[i] = inp
            video_input_depth[i] = input_depth
            video_input_mask[i] = input_mask
            video_gt_mask[i] = gt_mask
            if affordance_label == True :
                video_target_label[i] = torch.tensor([1.])
            else :
                video_target_label[i] = torch.tensor([0.])

            mask_path_list.append(mask_path)
            image_index_list.append(_index)
            gt_mask_path_list.append(gt_mask_path)

        # Meta info
        meta = {'index': index, 'mask_path_list': mask_path_list, 'image_index_list' : image_index_list, 'gt_mask_path_list' : gt_mask_path_list}
        
        return video_input, video_input_depth, video_input_mask, video_target_label, meta, video_gt_mask

    def random_shift(self, img, mask, depth, height_shift, width_shift):
        IMG_HEIGHT = self.inp_res
        IMG_WIDTH = self.inp_res

        shift_img = torch.zeros_like(img)
        shift_mask = torch.zeros_like(mask)
        shift_depth = torch.zeros_like(depth)

        if height_shift >= 0 and width_shift >= 0:
            shift_img[:, height_shift:IMG_HEIGHT, width_shift:IMG_WIDTH] = img[:, 0: (IMG_HEIGHT - height_shift), \
                0: (IMG_WIDTH - width_shift)]
            shift_mask[:, height_shift:IMG_HEIGHT, width_shift:IMG_WIDTH] = mask[:, 0: (IMG_HEIGHT - height_shift), \
                0: (IMG_WIDTH - width_shift)]
            shift_depth[:, height_shift:IMG_HEIGHT, width_shift:IMG_WIDTH] = depth[:, 0: (IMG_HEIGHT - height_shift), \
                0: (IMG_WIDTH - width_shift)]
        elif height_shift < 0 and width_shift >= 0:
            shift_img[:, 0:(IMG_HEIGHT + height_shift), width_shift:IMG_WIDTH] = img[:, -height_shift: IMG_HEIGHT, \
                0: (IMG_WIDTH - width_shift)]
            shift_mask[:, 0:(IMG_HEIGHT + height_shift), width_shift:IMG_WIDTH] = mask[:, -height_shift: IMG_HEIGHT, \
                0: (IMG_WIDTH - width_shift)]
            shift_depth[:, 0:(IMG_HEIGHT + height_shift), width_shift:IMG_WIDTH] = depth[:, -height_shift: IMG_HEIGHT, \
                0: (IMG_WIDTH - width_shift)]
        elif height_shift >= 0 and width_shift < 0:
            shift_img[:, height_shift:IMG_HEIGHT, 0:(IMG_WIDTH + width_shift)] = img[:, 0: (IMG_HEIGHT - height_shift), \
                -width_shift: IMG_WIDTH]
            shift_mask[:, height_shift:IMG_HEIGHT, 0:(IMG_WIDTH + width_shift)] = mask[:, 0: (IMG_HEIGHT - height_shift), \
                -width_shift: IMG_WIDTH]
            shift_depth[:, height_shift:IMG_HEIGHT, 0:(IMG_WIDTH + width_shift)] = depth[:, 0: (IMG_HEIGHT - height_shift), \
                -width_shift: IMG_WIDTH]
        elif height_shift < 0 and width_shift < 0:
            shift_img[:, 0:(IMG_HEIGHT + height_shift), 0:(IMG_WIDTH + width_shift)] = img[:, 0: (IMG_HEIGHT + height_shift), \
                0: (IMG_WIDTH + width_shift)]
            shift_mask[:, 0:(IMG_HEIGHT + height_shift), 0:(IMG_WIDTH + width_shift)] = mask[:, 0: (IMG_HEIGHT + height_shift), \
                0: (IMG_WIDTH + width_shift)]
            shift_depth[:, 0:(IMG_HEIGHT + height_shift), 0:(IMG_WIDTH + width_shift)] = depth[:, 0: (IMG_HEIGHT + height_shift), \
                0: (IMG_WIDTH + width_shift)]
        return shift_img, shift_mask, shift_depth

    def __len__(self):
        if self.is_train:
            return len(self.train_list)
        else:
            return len(self.valid_list)

def sad_eval(**kwargs):
    return Sad_eval(**kwargs)

sad_eval.njoints = 1  # ugly but works
