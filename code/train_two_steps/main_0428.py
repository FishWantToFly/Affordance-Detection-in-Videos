'''
Train step 1 on sad dataset.

# training 
python main_0428.py
# training using only 10 actions (for test)
python main.py -t 

# resume training from checkpoint
python main_0428.py --resume ./checkpoint_0605_coco_all_step_1/checkpoint_best_iou.pth.tar
# resume pre-training from checkpoint
python main_0428.py --resume ./checkpoint_0605_coco_all_step_1/checkpoint_best_iou.pth.tar -p

python main_0428.py --resume ./checkpoint_0612_8000_to_200/checkpoint_best_iou.pth.tar -p






# draw line chart (loss and IoU curve)
python main.py --resume ./checkpoint/checkpoint_20.pth.tar -w

# relabel train/test (visualize in same architecture)
python main_0428.py --resume ./checkpoint_0428/checkpoint_best_iou.pth.tar -e -r

python main_0428.py --resume ./checkpoint_0606_coco_fine_tune_step_1/checkpoint_best_iou.pth.tar -e -r

# temp
python main_0428.py --resume ./checkpoint_0606_coco_fine_tune_step_1/checkpoint_best_iou.pth.tar -e
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
from affordance.utils.imutils import batch_with_heatmap, sample_test, relabel_heatmap
from affordance.utils.transforms import fliplr, flip_back
import affordance.models as models
import affordance.datasets as datasets
import affordance.losses as losses

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
    plt.legend(['Val iou'], loc='upper left')
    plt.savefig(os.path.join(args.checkpoint, 'log_iou.png'))
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

    if args.pre_train:
        # pre train lr = 5e-4
        args.lr = 5e-5

    if args.resume != '' and args.pre_train == False:
        args.checkpoint = ('/').join(args.resume.split('/')[:2])
    if args.relabel == True:
        args.test_batch = 1
    if args.test == True:
        args.train_batch = 4
        args.test_batch = 4
        args.epochs = 20

    if args.evaluate and args.relabel == False:
        args.test_batch = 10

    # write line-chart and stop program
    if args.write: 
        draw_line_chart(args, os.path.join(args.checkpoint, 'log.txt'))
        return

    # idx is the index of joints used to compute accuracy
    # if args.dataset in ['mpii', 'lsp']:
    #     idx = [1,2,3,4,5,6,11,12,15,16]
    # elif args.dataset == 'coco':
    #     idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    # elif args.dataset == 'sad' or args.dataset == 'sad_step_1':
    #     idx = [1] # support affordance
    # else:
    #     print("Unknown dataset: {}".format(args.dataset))
    #     assert False
    idx = [1]

    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # create model
    njoints = datasets.__dict__[args.dataset].njoints

    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    model = models.__dict__[args.arch](num_stacks=args.stacks,
                                       num_blocks=args.blocks,
                                       num_classes=njoints,
                                       resnet_layers=args.resnet_layers)

    # 2020.6.7
    # freeze feature extraction and first one hg model paras
    # freeze_list = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, \
    #     model.hg[0], model.res[0], model.fc[0], \
    #     model.score[0],model.fc_[0], model.score_[0]]
    # for freeze_layer in freeze_list :
    #     for param in freeze_layer.parameters():
    #         param.requires_grad = False

    model = torch.nn.DataParallel(model).to(device)

    # define loss function (criterion) and optimizer
    criterion = losses.IoULoss().to(device)

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
        # optimizer = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, model.parameters()),
        #     lr=args.lr,
        # )
    else:
        print('Unknown solver: {}'.format(args.solver))
        assert False

    # optionally resume from a checkpoint
    title = args.dataset + ' ' + args.arch
    if args.pre_train:
        if isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)

                # start from epoch 0
                args.start_epoch = 0
                best_iou = 0
                model.load_state_dict(checkpoint['state_dict'])

                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
                logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
                logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Val IoU'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    elif args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            # best_iou = checkpoint['best_iou']

            # start from epoch 0
            args.start_epoch = 0
            best_iou = 0

            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
        
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Val IoU'])

    print('    Total params: %.2fM'
          % (sum(p.numel() for p in model.parameters())/1000000.0))

    # create data loader
    train_dataset = datasets.__dict__[args.dataset](is_train=True, **vars(args)) #-> depend on args.dataset to replace with datasets
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )

    '''
    for i, (input, input_depth, target, meta) in enumerate(train_loader):
        print(len(input))
        print(input[0].shape)
        print(input_depth[0].shape)
        print(target[0].shape)
        return
    '''
    

    val_dataset = datasets.__dict__[args.dataset](is_train=False, **vars(args))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # redraw training / test label :
    global RELABEL
    if args.relabel:
        RELABEL = True
        if args.evaluate:
            print('\nRelabel val label')
            loss, iou, predictions = validate(val_loader, model, criterion, njoints,
                                    args.checkpoint, args.debug, args.flip)
            # Because test and val are all considered -> iou is uesless
            # print("Val IoU: %.3f" % (iou))
            return 

    # evaluation only
    global JUST_EVALUATE
    JUST_EVALUATE = False
    if args.evaluate:
        print('\nEvaluation only')
        if args.debug :
            print('Draw pred /gt heatmap')
        JUST_EVALUATE = True
        loss, iou, predictions = validate(val_loader, model, criterion, njoints,
                                           args.checkpoint, args.debug, args.flip)
        print("Val IoU: %.3f" % (iou))
        return

    ## backup when training starts
    code_backup_dir = 'code_backup'
    mkdir_p(os.path.join(args.checkpoint, code_backup_dir))
    os.system('cp ../affordance/models/hourglass.py %s/%s/hourglass.py' % (args.checkpoint, code_backup_dir))
    os.system('cp ../affordance/datasets/sad.py %s/%s/sad.py' % (args.checkpoint, code_backup_dir))
    this_file_name = os.path.split(os.path.abspath(__file__))[1]
    os.system('cp ./%s %s' % (this_file_name, os.path.join(args.checkpoint, code_backup_dir, this_file_name)))

    # train and eval
    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):        
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # decay sigma
        if args.sigma_decay > 0:
            train_loader.dataset.sigma *=  args.sigma_decay
            val_loader.dataset.sigma *=  args.sigma_decay

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer,
                                      args.debug, args.flip)

        # evaluate on validation set
        valid_loss, valid_iou, predictions = validate(val_loader, model, criterion,
                                                  njoints, args.checkpoint, args.debug, args.flip)
        print("Val IoU: %.3f" % (valid_iou))

        # append logger file
        logger.append([epoch + 1, lr, train_loss, valid_loss, valid_iou])

        # remember best acc and save checkpoint
        is_best_iou = valid_iou > best_iou
        best_iou = max(valid_iou, best_iou)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_iou': best_iou,
            'optimizer' : optimizer.state_dict(),
        }, is_best_iou, checkpoint=args.checkpoint, snapshot=args.snapshot)

    logger.close()

    print("Best iou = %.3f" % (best_iou))
    # draw_line_chart(args, os.path.join(args.checkpoint, 'log.txt'))

def train(train_loader, model, criterion, optimizer, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    gt_win, pred_win = None, None
    bar = Bar('Train', max=len(train_loader))
    for i, (input, input_depth, target, meta) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input, input_depth, target = input.to(device), input_depth.to(device), target.to(device, non_blocking=True)
        # target_weight = meta['target_weight'].to(device, non_blocking=True)

        batch_size = input.shape[0]
        loss = 0
        # store first two stack feature # 2 = first and second stack, 6 = video length
        video_feature_cache = torch.zeros(batch_size, 6, 2, 256, output_res, output_res)
        
        with torch.no_grad():
            # first compute
            for j in range(6):
                input_now = input[:, j] # [B, 3, 256, 256]
                input_depth_now = input_depth[:, j]
                target_now = target[:, j]
                _, out_tsm_feature = model(torch.cat((input_now, input_depth_now), 1)) # [B, 4, 256, 256]
                for k in range(2):
                    video_feature_cache[:, j, k] = out_tsm_feature[k]

            # TSM module
            b, t, _, c, h, w = video_feature_cache.size()
            fold_div = 8
            fold = c // fold_div
            new_tsm_feature = torch.zeros(batch_size, 6, 2, 256, output_res, output_res)
            for j in range(2):
                x = video_feature_cache[:, :, j]
                temp = torch.zeros(batch_size, 6, 256, output_res, output_res)
                temp[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
                temp[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
                temp[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
                new_tsm_feature[:, :, j] = temp

        new_tsm_feature = new_tsm_feature.to(device)
        # compute use TSM feature
        for j in range(6):
            input_now = input[:, j] # [B, 3, 256, 256]
            input_depth_now = input_depth[:, j]
            target_now = target[:, j]
            output, _ = model(torch.cat((input_now, input_depth_now), 1), True, new_tsm_feature[:, j])

            if type(output) == list:  # multiple output # beacuse of intermediate prediction
                for o in output:
                    loss += criterion(o, target_now)
                output = output[-1]
            else:  # single output
                pass
                # loss = criterion(output, target, target_weight)
        
        # # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
                    batch=i + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    )
        bar.next()
    bar.finish()
    return losses.avg


def validate(val_loader, model, criterion, num_classes, checkpoint, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ioues = AverageMeter()

    # predictions
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)

    # switch to evaluate mode
    model.eval()

    gt_win, pred_win = None, None
    iou = None
    end = time.time()
    bar = Bar('Eval ', max=len(val_loader))
    with torch.no_grad():
        for i, (input, input_depth, target, meta) in enumerate(val_loader):
            # if RELABEL and i == 2 : break

            # measure data loading time
            data_time.update(time.time() - end)

            input = input.to(device, non_blocking=True)
            input_depth = input_depth.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            batch_size = input.shape[0]
            loss = 0
            iou_list = []

            # store first two stack feature # 2 = first and second stack, 6 = video length
            video_feature_cache = torch.zeros(batch_size, 6, 2, 256, output_res, output_res)

            # first compute
            for j in range(6):
                input_now = input[:, j] # [B, 3, 256, 256]
                input_depth_now = input_depth[:, j]
                target_now = target[:, j]
                _, out_tsm_feature = model(torch.cat((input_now, input_depth_now), 1)) # [B, 4, 256, 256]
                for k in range(2):
                    video_feature_cache[:, j, k] = out_tsm_feature[k]

            # TSM module
            b, t, _, c, h, w = video_feature_cache.size()
            fold_div = 8
            fold = c // fold_div
            new_tsm_feature = torch.zeros(batch_size, 6, 2, 256, output_res, output_res)
            for j in range(2):
                x = video_feature_cache[:, :, j]
                temp = torch.zeros(batch_size, 6, 256, output_res, output_res)
                temp[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
                temp[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
                temp[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
                new_tsm_feature[:, :, j] = temp

            new_tsm_feature = new_tsm_feature.to(device)
            # compute use TSM feature
            for j in range(6):
                input_now = input[:, j] # [B, 3, 256, 256]
                input_depth_now = input_depth[:, j]
                target_now = target[:, j]
                output, _ = model(torch.cat((input_now, input_depth_now), 1), True, new_tsm_feature[:, j])

                if type(output) == list:  # multiple output # beacuse of intermediate prediction
                    for o in output:
                        loss += criterion(o, target_now)
                    output = output[-1]
                else:  # single output
                    pass

                temp_iou = intersectionOverUnion(output.cpu(), target_now.cpu(), idx) # have not tested
                iou_list.append(temp_iou)
                score_map = output[-1].cpu() if type(output) == list else output.cpu()
            

                if RELABEL:
                    # save in same checkpoint
                    raw_mask_path = meta['mask_path_list'][j][0]
                    img_index = meta['image_index_list'][j][0]
                    temp_head = ('/').join(raw_mask_path.split('/')[:-8])
                    temp_tail = ('/').join(raw_mask_path.split('/')[-6:])
                    temp = os.path.join(temp_head, 'code/train_two_steps', checkpoint, 'pred_vis', temp_tail)
                    relabel_mask_dir, relabel_mask_name = os.path.split(temp)
                    relabel_mask_dir = os.path.dirname(relabel_mask_dir)

                    raw_mask_rgb_path = os.path.join(os.path.dirname(os.path.dirname(raw_mask_path)), 'first_mask_rgb', relabel_mask_name)
                    new_mask_rgb_path = os.path.join(relabel_mask_dir, 'gt_' + relabel_mask_name)
                    raw_rgb_frame_path = os.path.join(os.path.dirname(os.path.dirname(raw_mask_path)), 'raw_frames', \
                        relabel_mask_name[:-4] + '.png')

                    # print(relabel_mask_dir)
                    # print(relabel_mask_name)
                    from PIL import Image
                    import numpy as np
                    if os.path.exists(raw_mask_rgb_path):
                        gt_mask_rgb = np.array(Image.open(raw_mask_rgb_path))
                    else :
                        gt_mask_rgb = np.array(Image.open(raw_rgb_frame_path))
                    # print(input_now.shape)
                    # print(score_map.shape)
                    pred_batch_img, pred_mask = relabel_heatmap(input_now, score_map, 'pred') # return an Image object
                    
                    if not isdir(relabel_mask_dir):
                        mkdir_p(relabel_mask_dir)

                    if not gt_win or not pred_win:
                        ax1 = plt.subplot(121)
                        ax1.title.set_text('MASK_RGB_GT')
                        gt_win = plt.imshow(gt_mask_rgb)
                        ax2 = plt.subplot(122)
                        ax2.title.set_text('Mask_RGB_PRED')
                        pred_win = plt.imshow(pred_batch_img)
                    else:
                        gt_win.set_data(gt_mask_rgb)
                        pred_win.set_data(pred_batch_img)
                    plt.plot()
                    index_name = "%05d.jpg" % (img_index)
                    plt.savefig(os.path.join(relabel_mask_dir, 'vis_' + index_name))
                    pred_mask.save(os.path.join(relabel_mask_dir, index_name)) 
                    

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            # acces.update(acc[0], input.size(0))
            ioues.update(sum(iou_list) / len(iou_list), input.size(0))

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
    return losses.avg, ioues.avg, predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', metavar='DATASET', default='sad_step_1',
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
    # parser.add_argument('--inp-res', default=128, type=int,
    #                     help='input resolution (default: 256)')
    # parser.add_argument('--out-res', default=32, type=int,
    #                 help='output resolution (default: 64, to gen GT)')

                        
    parser.add_argument('--dataset-list-dir-path', default='/home/s5078345/Affordance-Detection-on-Video/dataset_two_steps/data_list', type=str,
                    help='dir of train/test data list')

    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
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
    parser.add_argument('--train-batch', default=8, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=8, type=int, metavar='N',
                        help='train batchsize')

    # 1 GPU setting
    # parser.add_argument('--train-batch', default=4, type=int, metavar='N',
    #                     help='train batchsize')
    # parser.add_argument('--test-batch', default=4, type=int, metavar='N',
    #                     help='train batchsize')


    parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[50, 80],
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

    parser.add_argument('-p', '--pre-train', action='store_true',
                        help='pre-train or not')

    main(parser.parse_args())
