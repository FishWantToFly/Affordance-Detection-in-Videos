'''
Evaluate for step 1 + step 2 
Step 1 output is pre-generated (from args.mask)

python main_eval.py --mask ./checkpoint_0428/pred_vis --resume ./checkpoint_0523_input_pred_mask/checkpoint_best_iou.pth.tar -e

python main_eval.py --mask ./checkpoint_0613_coco_sad_train/pred_vis --resume ./checkpoint_0616_stateless/checkpoint_best_iou.pth.tar -e
'''
from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

import _init_paths
from affordance import Bar
from affordance.utils.logger import Logger, savefig
from affordance.utils.evaluation import accuracy, AverageMeter, final_preds, intersectionOverUnion
from affordance.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from affordance.utils.osutils import mkdir_p, isfile, isdir, join
from affordance.utils.imutils import batch_with_heatmap, sample_test, relabel_heatmap, eval_heatmap
from affordance.utils.transforms import fliplr, flip_back
import affordance.models as models
import affordance.datasets as datasets
import affordance.losses as losses
from sklearn.metrics import accuracy_score





# get model names and dataset names
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


dataset_names = sorted(name for name in datasets.__dict__
    if name.islower() and not name.startswith("__")
    and callable(datasets.__dict__[name]))


# init global variables
best_iou = 0
output_res = None
idx = []

RELABEL = False

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # There is BN issue for early version of PyTorch
                        # see https://github.com/bearpaw/pytorch-pose/issues/33

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
    output_res = args.out_res

    # 2020.3.2
    global REDRAW

    # 2020.3.4
    # if you do type arg.resume
    # args.checkpoint would be derived from arg.resume
    if args.resume != '':
        args.checkpoint = ('/').join(args.resume.split('/')[:2])
        
    if args.relabel == True:
        args.test_batch = 1
    elif args.test == True:
        args.train_batch = 2
        args.test_batch = 2
        args.epochs = 10
    elif args.evaluate == True:
        args.test_batch = 10

    # write line-chart and stop program
    if args.write: 
        draw_line_chart(args, os.path.join(args.checkpoint, 'log.txt'))
        return

    # idx is the index of joints used to compute accuracy
    if args.dataset in ['mpii', 'lsp']:
        idx = [1,2,3,4,5,6,11,12,15,16]
    elif args.dataset == 'coco':
        idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    elif args.dataset == 'sad' or args.dataset == 'sad_step_2' or args.dataset == 'sad_step_2_eval' \
        or  args.dataset == 'sad_eval':
        idx = [1] # support affordance
    else:
        print("Unknown dataset: {}".format(args.dataset))
        assert False

    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # create model
    njoints = datasets.__dict__[args.dataset].njoints

    model = models.__dict__[args.arch](num_stacks=args.stacks,
                                       num_blocks=args.blocks,
                                       num_classes=njoints,
                                       resnet_layers=args.resnet_layers)


    model = torch.nn.DataParallel(model).to(device)

    # define loss function (criterion) and optimizer
    criterion = losses.BCELoss().to(device)

    if args.solver == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
    elif args.solver == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )
    else:
        print('Unknown solver: {}'.format(args.solver))
        assert False

    # optionally resume from a checkpoint
    title = args.dataset + ' ' + args.arch
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_iou = checkpoint['best_iou']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Val Acc'])

    print('    Total params: %.2fM'
          % (sum(p.numel() for p in model.parameters())/1000000.0))

    # create data loader
    train_dataset = datasets.__dict__[args.dataset](is_train=True, **vars(args)) #-> depend on args.dataset to replace with datasets
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )

    val_dataset = datasets.__dict__[args.dataset](is_train=False, **vars(args))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # for i, (input, input_depth, input_mask, target, meta, gt_mask) in enumerate(val_loader):
    #     if i == 10 : break
    #     print(gt_mask.shape)
    
    # return 


    
    # redraw training / test label :
    global RELABEL
    if args.relabel:
        RELABEL = True
        if args.evaluate:
            print('\nRelabel val label')
            loss, acc, final_acc  = validate(val_loader, model, criterion, njoints,
                                    args.checkpoint, args.debug, args.flip)
            print("Final acc: %.3f" % (final_acc))
            return 

    # evaluation only
    global JUST_EVALUATE
    JUST_EVALUATE = False
    if args.evaluate:
        print('\nEvaluation only')
        JUST_EVALUATE = True
        loss, acc, final_acc = validate(val_loader, model, criterion, njoints,
                                           args.checkpoint, args.debug, args.flip)
        # print("(Step 2) Val acc: %.3f" % (acc))
        print("Final acc: %.3f" % (final_acc))
        return
    

def validate(val_loader, model, criterion, num_classes, checkpoint, debug=False, flip=True):
    import numpy as np

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    ioues = AverageMeter()

    # for statistic
    gt_trues = AverageMeter()
    gt_falses = AverageMeter()
    pred_trues = AverageMeter() # true == true and iou > 50%
    pred_falses = AverageMeter()
    
    pred_trues_first = AverageMeter() # true == true

    # iou > 50% and step 2 labels are both right -> correcct
    # if label is false (and pred is false too) -> correct
    final_acces = AverageMeter() 

    # switch to evaluate mode
    model.eval()

    gt_win, pred_win = None, None
    iou = None
    end = time.time()
    bar = Bar('Eval ', max=len(val_loader))
    with torch.no_grad():
        for i, (input, input_depth, input_mask, target, meta, gt_mask) in enumerate(val_loader):
            
            # if i == 10 : break

            # measure data loading time
            data_time.update(time.time() - end)

            input, input_mask, target = input.to(device), input_mask.to(device), target.to(device, non_blocking=True)
            input_depth = input_depth.to(device)
            
            batch_size = input.shape[0]
            loss = 0
            last_state = None 
            acc_list = []
            iou_list = []
            final_acc_list = []

            # for statistic
            gt_true_list = []
            gt_false_list = []
            pred_true_list = []
            pred_false_list = []
            pred_true_first_list = []

            for j in range(6):
                input_now = input[:, j] # [B, 3, 256, 256]
                input_depth_now = input_depth[:, j]
                input_mask_now = input_mask[:, j]
                gt_mask_now = gt_mask[:, j]
                target_now = target[:, j]
                if j == 0:
                    output, last_state = model(torch.cat((input_now, input_depth_now, input_mask_now), 1))
                else : 
                    output, _ = model(torch.cat((input_now, input_depth_now, input_mask_now), 1), input_last_state = last_state)
                    # print(output.shape)

                round_output = torch.round(output).float()
                loss += criterion(output, target_now)

                temp_acc = float((round_output == target_now).sum()) / batch_size

                temp_1 = (round_output == 1) & (target_now == 1)
                temp_acc_1 = temp_1.cpu().numpy()
                temp_2 = (round_output == 0) & (target_now == 0)
                temp_acc_2 = temp_2.cpu().numpy()
                
                temp_iou = intersectionOverUnion(gt_mask_now, input_mask_now.cpu(), idx, return_list = True)
                
                final_pred_1 = np.logical_and(temp_acc_1, temp_iou > 0.5)
                final_pred_2 = temp_acc_2
                final_pred = np.logical_or(final_pred_1, final_pred_2)

                acc_list.append(temp_acc)
                final_acc_list.append(np.sum(final_pred) / batch_size)
                round_output = round_output.cpu()

                # for statistic
                temp_1 = (target_now == 1).cpu().numpy()
                temp_2 = (target_now == 0).cpu().numpy()
                gt_true_list.append(np.sum(temp_1) / batch_size)
                gt_false_list.append(np.sum(temp_2) / batch_size)

                pred_true_list.append(np.sum(final_pred_1) / batch_size)
                pred_false_list.append(np.sum(final_pred_2) / batch_size)
                pred_true_first_list.append(np.sum(temp_acc_1) / batch_size)

                
                if RELABEL:
                    '''
                    left image : GT
                        image : gt_mask_rgb
                        label : target_now
                    right image : predict result
                        image : raw_frames (pred false) or ./checkpoint_0428/pred_vis (pred true)
                        label : round_output
                    '''
                    from PIL import Image
                    import numpy as np
                    import copy

                    # save in same checkpoint
                    img_index = meta['image_index_list'][j][0]

                    raw_mask_path = meta['mask_path_list'][j][0]
                    gt_mask_path = meta['gt_mask_path_list'][j][0]

                    temp_head = ('/').join(gt_mask_path.split('/')[:-8])
                    temp_tail = ('/').join(gt_mask_path.split('/')[-5:])
                    temp = os.path.join(temp_head, 'code/train_two_steps/eval', 'pred_vis', temp_tail)
                    relabel_mask_dir, relabel_mask_name = os.path.split(temp)
                    relabel_mask_dir = os.path.dirname(relabel_mask_dir) # new dir name for pred_vis

                    # raw frame
                    raw_rgb_frame_path = os.path.join(os.path.dirname(os.path.dirname(gt_mask_path)), 'raw_frames', gt_mask_path.split('/')[-1][:-4] + '.png')
                    raw_frame = np.array(Image.open(raw_rgb_frame_path))

                    # gt_mask_rgb
                    gt_mask_rgb_path = os.path.join(os.path.dirname(os.path.dirname(gt_mask_path)), 'mask_rgb', gt_mask_path.split('/')[-1])
                    if os.path.exists(gt_mask_rgb_path):
                        gt_mask_rgb = np.array(Image.open(gt_mask_rgb_path))
                    else :
                        gt_mask_rgb = copy.deepcopy(raw_frame)

                    # pred mask
                    pred_mask_path = os.path.join(os.path.dirname(raw_mask_path), relabel_mask_name)
                    pred_mask = np.array(Image.open(pred_mask_path))

                    pred_mask_rgb = eval_heatmap(raw_frame, pred_mask) # generate rgb 


                    if not isdir(relabel_mask_dir):
                        mkdir_p(relabel_mask_dir)

                    gt_label_str = None 
                    pred_label_str = None
                    gt_output = gt_mask_rgb
                    pred_output = None

                    if target_now[0][0] == 0:
                        gt_label_str = "GT : False"
                    elif target_now[0][0] == 1:
                        gt_label_str = "GT : True"

                    if round_output[0][0] == 0:
                        pred_label_str = "Pred : False"
                        pred_output = raw_frame
                    elif round_output[0][0] == 1:
                        pred_output = pred_mask_rgb
                        if target_now[0][0] == 0 : 
                            pred_label_str = "Pred : True"
                        elif target_now[0][0] == 1 and temp_iou > 0.5 : 
                            pred_label_str = "Pred : True (IoU : O)"
                        elif target_now[0][0] == 1 and temp_iou <= 0.5 :
                            pred_label_str = "Pred : True (IoU : X)"
                        
                    # output_str = gt_label_str + '. ' + pred_label_str

                    if not gt_win or not pred_win:
                        ax1 = plt.subplot(121)
                        ax1.title.set_text(gt_label_str)
                        gt_win = plt.imshow(gt_output)
                        ax2 = plt.subplot(122)
                        ax2.title.set_text(pred_label_str)
                        pred_win = plt.imshow(pred_output)

                    else:
                        gt_win.set_data(gt_output)
                        pred_win.set_data(pred_output)

                        ax1.title.set_text(gt_label_str)
                        ax2.title.set_text(pred_label_str)

                    plt.plot()
                    index_name = "%05d.jpg" % (img_index)
                    plt.savefig(os.path.join(relabel_mask_dir, 'vis_' + index_name))
                    
                

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            acces.update(sum(acc_list) / len(acc_list), input.size(0))
            final_acces.update(sum(final_acc_list) / len(final_acc_list), input.size(0))

            # for statistic
            gt_trues.update(sum(gt_true_list) / len(gt_true_list), input.size(0))
            gt_falses.update(sum(gt_false_list) / len(gt_false_list), input.size(0))
            pred_trues.update(sum(pred_true_list) / len(pred_true_list), input.size(0))
            pred_falses.update(sum(pred_false_list) / len(pred_false_list), input.size(0))
            pred_trues_first.update(sum(pred_true_first_list) / len(pred_true_first_list), input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
                        batch=i + 1,
                        size=len(val_loader),
                        data=data_time.val,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg
                        )
            bar.next()
        bar.finish()
    
    # for statistic
    print("GT true : %.3f" % (gt_trues.avg))
    print("GT false : %.3f" % (gt_falses.avg))
    print("Pred true : %.3f" % (pred_trues.avg))
    print("Pred false : %.3f" % (pred_falses.avg))
    print("====")
    print("Pred true (no considering Iou) : %.3f" % (pred_trues_first.avg))
    print("IoU > 50 percent accuracy : %.3f" % (pred_trues.avg / pred_trues_first.avg))
    print()

    return losses.avg, acces.avg, final_acces.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', metavar='DATASET', default='sad_eval', ## different from sad_step_2
                        choices=dataset_names,
                        help='Datasets: ' +
                            ' | '.join(dataset_names) +
                            ' (default: mpii)')
    parser.add_argument('--image-path', default='/home/s5078345/Affordance-Detection-on-Video/dataset_two_steps', type=str,
                        help='path to images')
    parser.add_argument('--anno-path', default='', type=str,
                        help='path to annotation (json)')
    parser.add_argument('--year', default=2014, type=int, metavar='N',
                        help='year of coco dataset: 2014 (default) | 2017)')


    parser.add_argument('--inp-res', default=256, type=int,
                        help='input resolution (default: 256)')
    parser.add_argument('--out-res', default=64, type=int,
                    help='output resolution (default: 64, to gen GT)')
                        
    parser.add_argument('--dataset-list-dir-path', default='/home/s5078345/Affordance-Detection-on-Video/dataset_two_steps/data_list', type=str,
                    help='dir of train/test data list')

    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='ACNet',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: hg)')
    parser.add_argument('-s', '--stacks', default=4, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('--features', default=256, type=int, metavar='N',
                        help='Number of features in the hourglass')
    parser.add_argument('--resnet-layers', default=50, type=int, metavar='N',
                        help='Number of resnet layers',
                        choices=[18, 34, 50, 101, 152])
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    # Training strategy
    parser.add_argument('--solver', metavar='SOLVER', default='adam',
                        choices=['rms', 'adam'],
                        help='optimizers')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    # 2 GPU setting
    parser.add_argument('--train-batch', default=20, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=20, type=int, metavar='N',
                        help='train batchsize')

    parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--target-weight', dest='target_weight',
                        action='store_true',
                        help='Loss with target_weight')
    # Data processing
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')

    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--snapshot', default=20, type=int,
                        help='save models for every #snapshot epochs (default: 0)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')
    parser.add_argument('-w', '--write', dest='write', action='store_true',
                        help='wirte acc / loss curve')
    parser.add_argument('-t', '--test', dest='test', action='store_true',
                        help='Use all data or just 10 actions')

    # 2020.3.2 for relabel (only use once)
    parser.add_argument('-r', '--relabel', dest='relabel', action='store_true',
                        help='Use model prediction to relabel label')

    # 2020.5.20
    parser.add_argument('-m', '--mask',type=str, metavar='PATH',
                    help='input mask path')


    main(parser.parse_args())
