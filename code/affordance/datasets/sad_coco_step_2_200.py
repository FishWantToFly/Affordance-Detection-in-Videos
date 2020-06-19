'''
2020.6.16
Step 2 annotation for 200 coco dataset
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


class Sad_coco_step_2_200(data.Dataset):
    def __init__(self, is_train = True, **kwargs):
        self.img_folder = kwargs['image_path'] # root image folders
        self.is_train   = is_train # training set or test set
        self.inp_res    = kwargs['inp_res']
        self.out_res    = kwargs['out_res']
        self.sigma      = kwargs['sigma']
        self.dataset_list_dir_path = kwargs['dataset_list_dir_path']
        self.input_mask_dir = kwargs['mask']

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

        for action in action_list :
            temp = [] 
            action_rgb_frames = glob.glob(os.path.join(self.img_folder, action, image_dir_name, '*.png'))
            input_action_path = ('/').join(action.split('/')[2:])
            sorted_action_rgb_frames = sorted(action_rgb_frames)

            frame = sorted_action_rgb_frames[0] 
            frame_name = os.path.basename(frame)
            frame_dir_dir = os.path.dirname(os.path.dirname(frame))
            affordance_dir = os.path.basename(os.path.dirname(frame_dir_dir))

            print(frame)
            print(affordance_dir)

            depth = None

            ## USE GT MASK HERE
            input_mask = os.path.join(self.input_mask_dir, input_action_path, frame_name[:-4] + '.jpg')
            print(input_action_path)
            print(input_mask)
            print()

            affordance_label = None
            if affordance_dir == 'support_true' :
                affordance_label = True
            elif affordance_dir == 'support_false' :
                affordance_label = False 
            else :
                print("Mask got wrong QQ")

            _index = 0
            temp.append([frame, input_mask, depth, affordance_label, _index]) # use pred mask as input
            
            all_files.append(temp)

        return all_files
    
    def __getitem__(self, index):
        video_len = 1
        mask_path_list = []
        image_index_list = []

        # img_path, mask_path, depth_path = self.train_list[index]
        if self.is_train:
            video_data = self.train_list[index]
        else :
            video_data = self.valid_list[index]

        video_input = torch.zeros(video_len, 3, self.inp_res, self.inp_res)
        video_input_depth = torch.zeros(video_len, 1, self.inp_res, self.inp_res)
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
            # depth = load_depth(depth_path)
            
            nparts = 1 # should change if target is more than 2

            # Prepare image and groundtruth map
            # inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
            inp = resize(img, self.inp_res, self.inp_res) # get normalized rgb value
            # input_depth = resize(depth, self.inp_res, self.inp_res)
            input_depth = torch.zeros(1, self.inp_res, self.inp_res) # no meaning
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
        
        return video_input, video_input_depth, video_input_mask, video_target_label, meta

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

def sad_coco_step_2_200(**kwargs):
    return Sad_coco_step_2_200(**kwargs)

sad_coco_step_2_200.njoints = 1  # ugly but works
