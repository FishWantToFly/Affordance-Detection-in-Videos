'''
python output_curve.py --resume ./checkpoint/checkpoint_best_iou.pth.tar -w
'''

from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt
import random

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

import _init_paths


def draw_line_chart(args, log_read_dir):
    list_of_lists = []
    with open(log_read_dir) as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split('\t')]
            # in alternative, if you need to use the file content as numbers
            # inner_list = [int(elt.strip()) for elt in line.split(',')]
            list_of_lists.append(inner_list)
    epoch_idx_list, train_acc_list, val_acc_list = [], [], []
    train_loss_list, val_loss_list = [], []
    val_iou_list = []
    list_len = len(list_of_lists)
    for i in range(1, list_len):
        epoch_idx_list.append(i)
        train_loss_list.append(float(list_of_lists[i][2]))
        val_loss_list.append(float(list_of_lists[i][3]))
        val_iou_list.append(float(list_of_lists[i][4]))

    plt.xlabel('Epoch')
    plt.plot(epoch_idx_list, train_loss_list)
    plt.plot(epoch_idx_list, val_loss_list)
    plt.legend(['Train loss', 'Val loss'], loc='upper left')
    plt.savefig(os.path.join(args.checkpoint, 'log_loss.png'))
    plt.cla()

    plt.xlabel('Epoch')
    plt.plot(epoch_idx_list, val_iou_list)
    plt.legend(['Val acc'], loc='upper left')
    plt.savefig(os.path.join(args.checkpoint, 'log_acc.png'))
    plt.cla()

def main(args):
    global best_iou
    global idx
    global output_res

    # 2020.3.2
    global REDRAW

    # 2020.3.4
    # if you do type arg.resume
    # args.checkpoint would be derived from arg.resume
    if args.resume != '':
        args.checkpoint = ('/').join(args.resume.split('/')[:2])
        

    # write line-chart and stop program
    if args.write: 
        draw_line_chart(args, os.path.join(args.checkpoint, 'log.txt'))
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
 

    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-w', '--write', dest='write', action='store_true',
                        help='wirte acc / loss curve')


    main(parser.parse_args())
