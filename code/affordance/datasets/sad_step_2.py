'''
2020.5.4 two steps 
1. affordance segmentation (first_mask)
2. affordance binary classification
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


class Sad_step_2(data.Dataset):
    def __init__(self, is_train = True, **kwargs):
        self.img_folder = kwargs['image_path'] # root image folders
        self.is_train   = is_train # training set or test set
        self.inp_res    = kwargs['inp_res']
        self.out_res    = kwargs['out_res']
        self.sigma      = kwargs['sigma']
        self.dataset_list_dir_path = kwargs['dataset_list_dir_path']

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
        mask_dir_name = 'mask' # gt_mask_dir
        depth_dir_name = 'inpaint_depth'
        mask_rgb_dir_name = 'mask_rgb'
        for action in action_list :
            action_rgb_frames = glob.glob(os.path.join(self.img_folder, action, image_dir_name, '*.png'))
            semantic_name = os.path.basename(os.path.dirname(action)).split('_')[0]
            semantic_label = self.semantic_dict[semantic_name]

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
                    # mask = os.path.join(frame_dir_dir, mask_dir_name, frame_name[:-4] + '.jpg')
                    gt_mask = os.path.join(frame_dir_dir, mask_dir_name, frame_name[:-4] + '.jpg')
                    depth = os.path.join(frame_dir_dir, depth_dir_name, frame_name[:-4] + '.npy')

                    # frame_dir_dir : /home/s5078345/Affordance-Detection-on-Video/dataset_two_steps/./dataset_original/home_living_room/chair_2_angle_2/remove_object_on_it_1
                    # NEED TO FIND CLASSIFICATION GT FROM dataset_original mask and mask_rgb
                    # if mask_rgb None : false. if mask_rgb exists image : true
                    origin_dataset_name = 'dataset_original'
                    temp_1 = ('/').join(frame_dir_dir.split('/')[0:6])
                    temp_2 = ('/').join(frame_dir_dir.split('/')[7:])
                    origin_frame_dir_dir = os.path.join(temp_1, origin_dataset_name, temp_2)
                    mask_rgb = os.path.join(frame_dir_dir, mask_rgb_dir_name, frame_name[:-4] + '.jpg')

                    affordance_label = None
                    if os.path.exists(gt_mask) and os.path.exists(mask_rgb) :
                        affordance_label = True
                    elif os.path.exists(gt_mask) and not os.path.exists(mask_rgb) :
                        affordance_label = False 
                    else :
                        print("Mask got wrong QQ")

                    temp.append([frame, gt_mask, depth, affordance_label, _index])
                    
                all_files.append(temp)

                # reach the end
                if (start_frame + 6) == len(action_rgb_frames):
                    break
        return all_files
    
    def __getitem__(self, index):
        video_len = 6
        mask_path_list = []
        image_index_list = []

        # img_path, mask_path, depth_path = self.train_list[index]
        if self.is_train:
            video_data = self.train_list[index]
        else :
            video_data = self.valid_list[index]

        video_input = torch.zeros(video_len, 3, self.inp_res, self.inp_res)
        # video_input_depth = torch.zeros(video_len, 1, self.inp_res, self.inp_res)
        video_input_mask = torch.zeros(video_len, 1, self.inp_res, self.inp_res)
        video_target_label = torch.zeros(video_len, 1)

        # Occlusion Preprocess (same occlusion for one action)
        random.seed()
        prob_keep_image = random.random()
        KEEP_IMAGE = prob_keep_image > 0.8

        G, S, S_num = self.inp_res, 32, int (self.inp_res / 32) # S_num = G / S
        occlusion_map = torch.zeros(G, G) # 1 represents save image, 0 represents occluded
        if KEEP_IMAGE == False :
            for i in range(S_num):
                for j in range(S_num):
                    prob_keep_patch = random.random()
                    if prob_keep_patch > 0.25 :
                        occlusion_map[i][j] = 1

        # Random shift preprocess
        random_shift_prob = random.random()
        h_num = int (self.inp_res / 6)
        height_shift = random.randint(-h_num, h_num)
        w_num = int (self.inp_res / 6)
        width_shift = random.randint(-w_num, w_num)

        for i in range(video_len):
            img_path, mask_path, depth_path, affordance_label, _index = video_data[i]

            # load image and mask
            img = load_image(img_path)  # CxHxW
            a = load_mask(mask_path)    # 1xHxW
            depth = load_depth(depth_path)
            
            nparts = 1 # should change if target is more than 2

            # Prepare image and groundtruth map
            # inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
            inp = resize(img, self.inp_res, self.inp_res) # get normalized rgb value
            input_depth = resize(depth, self.inp_res, self.inp_res)
            # Generate input mask
            input_mask = torch.zeros(nparts, self.inp_res, self.inp_res) # [1, out_res, out_res]
            _input_mask = a[0] # HxW
            _input_mask = resize(_input_mask, self.inp_res, self.inp_res) # [inp_res, inp_res]


            ######################################################
            # Data augmentation 
            # Occlusion (Stochastic Cutout occlusions)
            if self.is_train == True and KEEP_IMAGE == False :
                for x in range(S_num):
                    for y in range(S_num):
                        if occlusion_map[x][y] == 0 :
                            inp[:, x*S : x*S + S, y*S : y*S + S] = 0
                            input_depth[x*S : x*S + S, y*S : y*S + S] = 0
                            _input_mask[x*S : x*S + S, y*S : y*S + S] = 0 

            # Random shift
            if self.is_train == True :
                if random_shift_prob > 0.5 :
                    inp, _input_mask, input_depth = self.random_shift(inp, _input_mask, input_depth, h_num, w_num)
            input_mask[0] = _input_mask
            ##############################################
            # Output
            video_input[i] = inp
            # video_input_depth[i] = input_depth
            video_input_mask[i] = input_mask
            if affordance_label == True :
                video_target_label[i] = torch.tensor([1.])
            else :
                video_target_label[i] = torch.tensor([0.])

            mask_path_list.append(mask_path)
            image_index_list.append(_index)

        # Meta info
        meta = {'index': index, 'mask_path_list': mask_path_list, 'image_index_list' : image_index_list}
        
        return video_input, video_input_mask, video_target_label, meta

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

def sad_step_2(**kwargs):
    return Sad_step_2(**kwargs)

sad_step_2.njoints = 1  # ugly but works
