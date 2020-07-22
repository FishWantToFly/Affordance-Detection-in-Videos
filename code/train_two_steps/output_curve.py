'''
python output_curve.py --resume ./checkpoint/checkpoint_0604_coco_step_1.pth.tar -w

python output_curve.py --resume ./checkpoint_0714_final/checkpoint_0604_coco_step_1.pth.tar -w
python output_curve.py --resume ./checkpoint/checkpoint_0604_coco_step_1.pth.tar -w
'''

from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import random

import _init_paths


# for step 1
def draw_line_chart(args, log_read_dir):
    list_of_lists = []
    with open(log_read_dir) as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split('\t')]
            # in alternative, if you need to use the file content as numbers
            # inner_list = [int(elt.strip()) for elt in line.split(',')]
            list_of_lists.append(inner_list)
    epoch_idx_list, train_att_loss_list, val_att_loss_list, val_att_iou_list = [], [], [], []
    train_region_loss_list, val_region_loss_list, val_region_iou_list = [], [], []
    train_existence_loss_list, val_existence_loss_list, val_existence_acc_list = [], [], []
    val_final_acc_list = []

    list_len = len(list_of_lists)
    for i in range(1, list_len):
        epoch_idx_list.append(i)
        train_att_loss_list.append(float(list_of_lists[i][2]))
        val_att_loss_list.append(float(list_of_lists[i][3]))
        val_att_iou_list.append(float(list_of_lists[i][4]))
        
        train_region_loss_list.append(float(list_of_lists[i][5]))
        val_region_loss_list.append(float(list_of_lists[i][6]))
        val_region_iou_list.append(float(list_of_lists[i][7]))

        train_existence_loss_list.append(float(list_of_lists[i][8]))
        val_existence_loss_list.append(float(list_of_lists[i][9]))
        val_existence_acc_list.append(float(list_of_lists[i][10]))
        
        val_final_acc_list.append(float(list_of_lists[i][11]))

    # Attention
    plt.xlabel('Epoch')
    plt.plot(epoch_idx_list, train_att_loss_list)
    plt.plot(epoch_idx_list, val_att_loss_list)
    plt.legend(['Train Attention Loss', 'Val Attention Loss'], loc='upper left')
    plt.savefig(os.path.join(args.checkpoint, 'log_attention_loss.png'))
    plt.cla()

    plt.xlabel('Epoch')
    plt.plot(epoch_idx_list, val_att_iou_list)
    plt.legend(['Val Attention IoU'], loc='upper left')
    plt.savefig(os.path.join(args.checkpoint, 'log_attention_iou.png'))
    plt.cla()

    # Region
    plt.xlabel('Epoch')
    plt.plot(epoch_idx_list, train_region_loss_list)
    plt.plot(epoch_idx_list, val_region_loss_list)
    plt.legend(['Train Region Loss', 'Val Region Loss'], loc='upper left')
    plt.savefig(os.path.join(args.checkpoint, 'log_region_loss.png'))
    plt.cla()

    plt.xlabel('Epoch')
    plt.plot(epoch_idx_list, val_region_iou_list)
    plt.legend(['Val Region IoU'], loc='upper left')
    plt.savefig(os.path.join(args.checkpoint, 'log_region_iou.png'))
    plt.cla()

    # Label
    plt.xlabel('Epoch')
    plt.plot(epoch_idx_list, train_existence_loss_list)
    plt.plot(epoch_idx_list, val_existence_loss_list)
    plt.legend(['Train Existence Loss', 'Val Existence Loss'], loc='upper left')
    plt.savefig(os.path.join(args.checkpoint, 'log_existence_loss.png'))
    plt.cla()

    plt.xlabel('Epoch')
    plt.plot(epoch_idx_list, val_existence_acc_list)
    plt.legend(['Val Existence Acc'], loc='upper left')
    plt.savefig(os.path.join(args.checkpoint, 'log_existence_acc.png'))
    plt.cla()

    # Final evaluation
    plt.xlabel('Epoch')
    plt.plot(epoch_idx_list, val_final_acc_list)
    plt.legend(['Val Fianl Acc'], loc='upper left')
    plt.savefig(os.path.join(args.checkpoint, 'log_final_acc.png'))
    plt.cla()

# # for step 1
# def draw_line_chart(args, log_read_dir):
#     list_of_lists = []
#     with open(log_read_dir) as f:
#         for line in f:
#             inner_list = [elt.strip() for elt in line.split('\t')]
#             # in alternative, if you need to use the file content as numbers
#             # inner_list = [int(elt.strip()) for elt in line.split(',')]
#             list_of_lists.append(inner_list)
#     epoch_idx_list, train_acc_list, val_acc_list = [], [], []
#     train_loss_list, val_loss_list = [], []
#     val_iou_list = []
#     list_len = len(list_of_lists)
#     for i in range(1, list_len):
#         epoch_idx_list.append(i)
#         train_loss_list.append(float(list_of_lists[i][2]))
#         val_loss_list.append(float(list_of_lists[i][3]))
#         val_iou_list.append(float(list_of_lists[i][4]))

#     plt.xlabel('Epoch')
#     plt.plot(epoch_idx_list, train_loss_list)
#     plt.plot(epoch_idx_list, val_loss_list)
#     plt.legend(['Train loss', 'Val loss'], loc='upper left')
#     plt.savefig(os.path.join(args.checkpoint, 'log_loss.png'))
#     plt.cla()

#     plt.xlabel('Epoch')
#     plt.plot(epoch_idx_list, val_iou_list)
#     plt.legend(['Val iou'], loc='upper left')
#     plt.savefig(os.path.join(args.checkpoint, 'log_iou.png'))
#     plt.cla()

# for step 2
# def draw_line_chart(args, log_read_dir):
#     list_of_lists = []
#     with open(log_read_dir) as f:
#         for line in f:
#             inner_list = [elt.strip() for elt in line.split('\t')]
#             # in alternative, if you need to use the file content as numbers
#             # inner_list = [int(elt.strip()) for elt in line.split(',')]
#             list_of_lists.append(inner_list)
#     epoch_idx_list, train_acc_list, val_acc_list = [], [], []
#     train_loss_list, val_loss_list = [], []
#     val_iou_list = []
#     list_len = len(list_of_lists)
#     for i in range(1, list_len):
#         epoch_idx_list.append(i)
#         train_loss_list.append(float(list_of_lists[i][2]))
#         val_loss_list.append(float(list_of_lists[i][3]))
#         val_iou_list.append(float(list_of_lists[i][4]))

#     plt.xlabel('Epoch')
#     plt.plot(epoch_idx_list, train_loss_list)
#     plt.plot(epoch_idx_list, val_loss_list)
#     plt.legend(['Train loss', 'Val loss'], loc='upper left')
#     plt.savefig(os.path.join(args.checkpoint, 'log_loss.png'))
#     plt.cla()

#     plt.xlabel('Epoch')
#     plt.plot(epoch_idx_list, val_iou_list)
#     plt.legend(['Val acc'], loc='upper left')
#     plt.savefig(os.path.join(args.checkpoint, 'log_acc.png'))
#     plt.cla()

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
