'''
Support affordance dataset (sad)
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

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *


class Sad(data.Dataset):
    def __init__(self, is_train = True, **kwargs):
        self.img_folder = kwargs['image_path'] # root image folders
        # self.jsonf:ile   = kwargs['anno_path']
        self.is_train   = is_train # training set or test set
        self.inp_res    = kwargs['inp_res']
        self.out_res    = kwargs['out_res']
        self.sigma      = kwargs['sigma']
        # self.scale_factor = kwargs['scale_factor']
        # self.rot_factor = kwargs['rot_factor']
        # self.label_type = kwargs['label_type']

        self.dataset_list_dir_path = kwargs['dataset_list_dir_path']
        # contain img and annotation
        if kwargs['test'] == False:
            self.train_list = self.load_full_file_list('train')
            self.valid_list = self.load_full_file_list('test')
        else :
            self.train_list = self.load_full_file_list('train_10')
            self.valid_list = self.load_full_file_list('test_10')


    def load_full_file_list(self, mode):
        all_files = []
        action_list = []
        read_action_list_dir = os.path.join(self.dataset_list_dir_path, mode + '.txt')
        with open(read_action_list_dir) as f:
            for line in f:
                inner_list = [elt.strip() for elt in line.split(' ')]
                action_list.append(inner_list[0])

        for action in action_list :
            action_rgb_frames = glob.glob(os.path.join(self.img_folder, action, 'raw_frames', '*.png'))
            for frame in action_rgb_frames :
                frame_name = os.path.basename(frame)
                frame_dir_dir = os.path.dirname(os.path.dirname(frame))
                mask = os.path.join(frame_dir_dir, 'mask', frame_name[:-4] + '.jpg')
                all_files.append([frame, mask])
                # print(frame)
                # print(mask)
                # print()
        return all_files

    
    def __getitem__(self, index):
        if self.is_train:
            img_path, mask_path = self.train_list[index]
        else    :
            img_path, mask_path = self.valid_list[index]

        # load image and mask
        img = load_image(img_path)  # CxHxW
        a = load_mask(mask_path)    # 1xHxW
        
        # pts = torch.Tensor(a['joint_self']) # [16, 3]
        # pts[:, 0:2] -= 1  # Convert pts to zero based

        # c = torch.Tensor(a['objpos']) # [2]
        # s = a['scale_provided'] # The scale keeps the height of the person as about 200 px.

        # For single-person pose estimation with a centered/scaled figure
        nparts = a.size(0) # [1]

        # Prepare image and groundtruth map
        # inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        inp = resize(img, self.inp_res, self.inp_res) # get normalized rgb value


        # Generate ground truth
        # tpts = a.clone() # target points [16, 3]
        target = torch.zeros(nparts, self.out_res, self.out_res) # [1, out_res, out_res]
        # target_weight = tpts[:, 2].clone().view(nparts, 1) # [16, 1] value is 0 or 1
        target_weight = torch.ones(1, 1) # [nparts, 1]

        for i in range(nparts):
            # if tpts[i, 1] > 0:
            #     tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2]+1, c, s, [self.out_res, self.out_res], rot=r))
            #     target[i], vis = draw_labelmap(target[i], tpts[i]-1, self.sigma, type=self.label_type)
            #     target_weight[i, 0] *= vis
            
            # set all target weight to 1 (actually we have only 1 target)

            target[i] = resize(a[i], self.out_res, self.out_res)
            pass

        # Meta info
        # meta = {'index' : index, 'pts' : pts, 'tpts' : tpts, 'target_weight': target_weight}
        meta = {'index' : index, 'target_weight': target_weight}
        
        return inp, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train_list)
        else:
            return len(self.valid_list)

def sad(**kwargs):
    return Sad(**kwargs)

sad.njoints = 1  # ugly but works
