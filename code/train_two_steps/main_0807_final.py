'''
2020.8.7
predicted attention heatmap as feature, feed into affordance label prediction branch

Use resnet-50 to extract feature (as feature extraction)

Model : hg_resnet_v2
Dataset : sad_resnet

# training 
python main_0807_final.py
# training using only 10 actions (for test)
python main_0807_final.py -t

# resume training from checkpoint
python main_0807_final.py --resume ./checkpoint_0605_coco_all_step_1/checkpoint_best_iou.pth.tar
# resume pre-training from checkpoint
python main_0807_final.py --resume ./checkpoint_0605_coco_all_step_1/checkpoint_best_iou.pth.tar -p

python main_0807_final.py --resume ./checkpoint_0720_ablation_5/checkpoint_5.pth.tar -e -r

'''
from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

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
best_final_acc = 0
output_res = None
idx = []

RELABEL = False

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # There is BN issue for early version of PyTorch
                        # see https://github.com/bearpaw/pytorch-pose/issues/33

def main(args):
    global best_final_acc
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
        args.epochs = 10

    if args.evaluate and args.relabel == False:
        args.test_batch = 4

    # write line-chart and stop program
    if args.write: 
        draw_line_chart(args, os.path.join(args.checkpoint, 'log.txt'))
        return

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



    model = torch.nn.DataParallel(model).to(device)

    # define loss function (criterion) and optimizer
    criterion_iou = losses.IoULoss().to(device)
    criterion_bce = losses.BCELoss().to(device)
    criterions = [criterion_iou, criterion_bce]

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
    if args.pre_train:
        if isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)

                # start from epoch 0
                args.start_epoch = 0
                best_final_acc = 0
                model.load_state_dict(checkpoint['state_dict'])

                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
                logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
                logger.set_names(['Epoch', 'LR', 'Train Attention Loss', 'Val Attention Loss', 'Val Attention Loss', \
                    'Val Region IoU', \
                    'Train Existence Acc', 'Val Existence Loss', 'Val Existence Acc', 'Val final acc'])

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
            best_final_acc = 0

            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
        
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Attention Loss', 'Val Attention Loss', 'Val Attention IoU', \
            'Val Region IoU', \
            'Train Existence Acc', 'Val Existence Loss', 'Val Existence Acc', 'Val final acc'])

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
    for i, (input, input_depth, target_heatmap, target_mask, target_label, meta) in enumerate(train_loader):
        print(len(input))
        print(input[0].shape)
        print(input_depth[0].shape)
        print(target_heatmap[0].shape)
        print(target_mask[0].shape)
        print(target_label[0].shape)
        return
    '''
    

    val_dataset = datasets.__dict__[args.dataset](is_train=False, **vars(args))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # # redraw training / test label :
    # global RELABEL
    # if args.relabel:
    #     RELABEL = True
    #     if args.evaluate:
    #         print('\nRelabel val label')
    #         val_att_loss, val_att_iou, \
    #             val_existence_loss, val_existence_acc , val_final_acc
    #                 = validate(val_loader, model, criterions, njoints, args.checkpoint, args.debug, args.flip)
    #         print("Val final acc: %.3f" % (val_final_acc))
    #         # Because test and val are all considered -> iou is uesless
    #         # print("Val IoU: %.3f" % (iou))
    #         return 

    # # evaluation only
    # global JUST_EVALUATE
    # JUST_EVALUATE = False
    # if args.evaluate:
    #     print('\nEvaluation only')
    #     JUST_EVALUATE = True
    #     val_att_loss, val_att_iou, \
    #         val_existence_loss, val_existence_acc , val_final_acc \
    #             = validate(val_loader, model, criterions, njoints, args.checkpoint, args.debug, args.flip)
    #     print("Val final acc: %.3f" % (val_final_acc))
    #     # print( val_att_loss, val_att_iou, val_region_loss, val_region_iou, \
    #     #     val_existence_loss, val_existence_acc , val_final_acc)

    #     return

    ## backup when training starts
    code_backup_dir = 'code_backup'
    mkdir_p(os.path.join(args.checkpoint, code_backup_dir))
    os.system('cp ../affordance/models/hourglass_resnet_v2.py %s/%s/hourglass_resnet_v2.py' % (args.checkpoint, code_backup_dir))
    os.system('cp ../affordance/datasets/sad_attention.py %s/%s/sad_attention.py' % (args.checkpoint, code_backup_dir))
    this_file_name = os.path.split(os.path.abspath(__file__))[1]
    os.system('cp ./%s %s' % (this_file_name, os.path.join(args.checkpoint, code_backup_dir, this_file_name)))

    # train and eval
    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # decay sigma
        if args.sigma_decay > 0:
            train_loader.dataset.sigma *= args.sigma_decay
            val_loader.dataset.sigma *= args.sigma_decay

        # train for one epoch
        train_att_loss, train_existence_loss \
            = train(train_loader, model, criterions, optimizer, args.debug, args.flip)

        # evaluate on validation set
        val_att_loss, val_att_iou, val_region_iou, \
            val_existence_loss, val_existence_acc , val_final_acc \
                = validate(val_loader, model, criterions, njoints, args.checkpoint, args.debug, args.flip)
        print("Val region IoU: %.3f" % (val_region_iou))
        print("Val label acc: %.3f" % (val_existence_acc))
        val_final_acc = val_region_iou + val_existence_acc

        # append logger file
        logger.append([epoch + 1, lr, train_att_loss, val_att_loss, val_att_iou, \
            val_region_iou, \
            train_existence_loss, val_existence_loss, val_existence_acc, 
            val_final_acc])

        # remember best acc and save checkpoint
        is_best_acc = val_final_acc > best_final_acc
        best_final_acc = max(val_final_acc, best_final_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_iou': best_final_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best_acc, checkpoint=args.checkpoint, snapshot=args.snapshot)

    logger.close()

    print("Best val final acc = %.3f" % (best_final_acc))

def train(train_loader, model, criterions, optimizer, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter()
    heatmap_losses = AverageMeter()
    mask_losses = AverageMeter()
    label_losses = AverageMeter()

    # Loss 
    criterion_iou, criterion_bce = criterions

    # switch to train mode
    model.train()

    end = time.time()

    gt_win, pred_win = None, None
    bar = Bar('Train', max=len(train_loader))
    for i, (input, input_depth, target_heatmap, target_mask, target_label, meta) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input, input_depth = input.to(device), input_depth.to(device)
        target_heatmap, target_mask, target_label = target_heatmap.to(device, non_blocking=True), target_mask.to(device, non_blocking=True), \
            target_label.to(device, non_blocking=True)

        batch_size = input.shape[0]
        total_loss, heatmap_loss, mask_loss, label_loss = 0, 0, 0, 0
        last_state = None
        last_tsm_buffer = None

        for j in range(6):
            input_now = input[:, j] # [B, 3, 256, 256]
            input_depth_now = input_depth[:, j] # [B, 1, 256, 256]
            target_heatmap_now = target_heatmap[:, j] # [B, 1, 64, 64]
            target_mask_now = target_mask[:, j] # [B, 1, 64, 64]
            target_label_now = target_label[:, j] # [B, 1]

            if j == 0:
                output_heatmap, output_label, output_state, output_tsm = model(input_now)
            else :
                output_heatmap, output_label, output_state, output_tsm = model(input_now, \
                    input_state = last_state, tsm_input = last_tsm_buffer)

            last_state = output_state
            last_tsm_buffer = output_tsm

            # Loss computation
            for o_heatmap in output_heatmap:
                temp = criterion_iou(o_heatmap, target_heatmap_now) * 0.3 + criterion_bce(o_heatmap, target_heatmap_now) * 0.3 # test now
                total_loss += temp
                heatmap_loss += temp
            # for o_mask in output_mask:
            #     temp = criterion_iou(o_mask, target_mask_now)
            #     total_loss += temp
            #     mask_loss += temp
            temp = criterion_bce(output_label, target_label_now)
            total_loss += temp
            label_loss += temp

        # measure accuracy and record loss
        total_losses.update(total_loss.item(), input.size(0))
        heatmap_losses.update(heatmap_loss.item(), input.size(0))
        # mask_losses.update(mask_loss.item(), input.size(0))
        label_losses.update(label_loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
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
                    loss=total_losses.avg,
                    )
        bar.next()
    bar.finish()
    return heatmap_losses.avg, label_losses.avg


def validate(val_loader, model, criterions, num_classes, checkpoint, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter()
    heatmap_losses = AverageMeter()
    mask_losses = AverageMeter()
    label_losses = AverageMeter()

    heatmap_ioues = AverageMeter()
    mask_ioues = AverageMeter()
    label_acces = AverageMeter()

    # iou > 50% and step 2 labels are both right -> correcct
    # if label is false (and pred is false too) -> correct
    final_acces = AverageMeter() 

    # for statistic
    gt_trues = AverageMeter() # positive
    gt_falses = AverageMeter() # negative
    pred_trues = AverageMeter() # true == true and iou > 50%
    pred_falses = AverageMeter()
    pred_trues_first = AverageMeter() # true == true

    # Loss
    criterion_iou, criterion_bce = criterions

    # switch to evaluate mode
    model.eval()

    gt_win, pred_win = None, None
    iou = None
    end = time.time()
    bar = Bar('Eval ', max=len(val_loader))
    with torch.no_grad():
        for i, (input, input_depth, target_heatmap, target_mask, target_label, meta) in enumerate(val_loader):
            # if RELABEL and i == 10 : break

            # measure data loading time
            data_time.update(time.time() - end)

            input, input_depth = input.to(device), input_depth.to(device)
            target_heatmap, target_mask, target_label = target_heatmap.to(device, non_blocking=True), target_mask.to(device, non_blocking=True), \
                target_label.to(device, non_blocking=True)

            batch_size = input.shape[0]
            total_loss, heatmap_loss, mask_loss, label_loss = 0, 0, 0, 0
            last_state = None
            last_tsm_buffer = None
            heatmap_iou_list, mask_iou_list, label_acc_list = [], [], []
            final_acc_list = []

            # for statistic
            gt_true_list, gt_false_list, pred_true_list, pred_false_list, pred_true_first_list = [], [], [], [], []

            for j in range(6):
                input_now = input[:, j] # [B, 3, 256, 256]
                input_depth_now = input_depth[:, j] # [B, 1, 256, 256]
                target_heatmap_now = target_heatmap[:, j] # [B, 1, 64, 64]
                target_mask_now = target_mask[:, j] # [B, 1, 64, 64]
                target_label_now = target_label[:, j] # [B, 1]

                if j == 0:
                    output_heatmap, output_label, output_state, output_tsm = model(input_now)
                else :
                    output_heatmap, output_label, output_state, output_tsm = model(input_now, \
                        input_state = last_state, tsm_input = last_tsm_buffer)
                last_state = output_state
                last_tsm_buffer = output_tsm

                # temp = output_heatmap[-1]
                # print(temp[temp > 0.5])

                ## if label predict negative : make that mask all black
                # round_output_label = torch.round(output_label).float() # [B, 1]
                # for o_mask in output_mask:
                #     o_mask[round_output_label == 0] = 0

                # Loss computation
                for o_heatmap in output_heatmap:
                    temp = criterion_iou(o_heatmap, target_heatmap_now) * 0.3 + criterion_bce(o_heatmap, target_heatmap_now) * 0.3
                    total_loss += temp
                    heatmap_loss += temp
                # for o_mask in output_mask:
                #     temp = criterion_iou(o_mask, target_mask_now)
                #     total_loss += temp
                #     mask_loss += temp
                temp = criterion_bce(output_label, target_label_now)
                total_loss += temp
                label_loss += temp

                # choose last one as prediction
                output_heatmap = output_heatmap[-1]
                # output_mask = output_mask[-1]

                ## Generate pred mask from pred heatmap
                round_output_label = torch.round(output_label).float() # [B, 1]
                output_mask = torch.zeros((batch_size, 1, 64, 64)).cuda()
                for _batch in range(batch_size) :
                    if round_output_label[_batch, 0] == 1:
                        output_mask[_batch] = output_heatmap[_batch]


                # evaluation metric
                heatmap_iou = intersectionOverUnion(output_heatmap.cpu(), target_heatmap_now.cpu(), idx) # experiemnt
                heatmap_iou_list.append(heatmap_iou)
                mask_iou = intersectionOverUnion(output_mask.cpu(), target_mask_now.cpu(), idx, return_list = True)
                mask_iou_list.append((sum(mask_iou) / len(mask_iou))[0])

                round_output_label = torch.round(output_label).float()
                label_acc = float((round_output_label == target_label_now).sum()) / batch_size
                label_acc_list.append(label_acc)
                
                score_map_mask = output_mask.cpu()
                # score_map_mask = output_heatmap.cpu()
                
                #########################
                # final evuation accuracy
                import numpy as np
                temp_1 = (round_output_label == 1) & (target_label_now == 1) # positve label correct
                temp_acc_1 = temp_1.cpu().numpy()
                temp_2 = (round_output_label == 0) & (target_label_now == 0) # negative label correct
                temp_acc_2 = temp_2.cpu().numpy()
                
                final_pred_1 = np.logical_and(temp_acc_1, mask_iou > 0.5) # positve label correct + iou > 50%
                final_pred_2 = temp_acc_2 # negative label correct
                final_acc = np.logical_or(final_pred_1, final_pred_2)
                final_acc_list.append(np.sum(final_acc) / batch_size)

                # for statistic
                temp_1 = (target_label_now == 1).cpu().numpy()
                temp_2 = (target_label_now == 0).cpu().numpy()
                gt_true_list.append(np.sum(temp_1) / batch_size)
                gt_false_list.append(np.sum(temp_2) / batch_size)

                pred_true_list.append(np.sum(final_pred_1) / batch_size)
                pred_false_list.append(np.sum(final_pred_2) / batch_size)
                pred_true_first_list.append(np.sum(temp_acc_1) / batch_size)
                ###############################

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
                    pred_batch_img, pred_mask = relabel_heatmap(input_now.cpu(), score_map_mask, 'pred') #
                    
                    if not isdir(relabel_mask_dir):
                        mkdir_p(relabel_mask_dir)

                    gt_label_str = None 
                    pred_label_str = None
                    from PIL import Image
                    pred_Image = Image.fromarray(pred_batch_img)

                    if target_label_now[0][0] == 0:
                        gt_label_str = "GT : False"
                    elif target_label_now[0][0] == 1:
                        gt_label_str = "GT : True"

                    if round_output_label[0][0] == 0:
                        pred_label_str = "Pred : False"
                    elif round_output_label[0][0] == 1:
                        pred_label_str = "Pred : True"

                    if not gt_win or not pred_win:
                        ax1 = plt.subplot(121)
                        ax1.title.set_text(gt_label_str)
                        gt_win = plt.imshow(gt_mask_rgb)
                        ax2 = plt.subplot(122)
                        ax2.title.set_text(pred_label_str)
                        pred_win = plt.imshow(pred_batch_img)
                    else:
                        gt_win.set_data(gt_mask_rgb)
                        pred_win.set_data(pred_batch_img)
                        ax1.title.set_text(gt_label_str)
                        ax2.title.set_text(pred_label_str)
                    plt.plot()
                    index_name = "%05d.jpg" % (img_index)
                    plt.savefig(os.path.join(relabel_mask_dir, 'vis_' + index_name))
                    pred_mask.save(os.path.join(relabel_mask_dir, index_name))
                    pred_Image.save(os.path.join(relabel_mask_dir, 'image_' + index_name))
                    # print(relabel_mask_dir)
                    
            # record final acc
            final_acces.update(sum(final_acc_list) / len(final_acc_list), input.size(0))

            # for statistic
            gt_trues.update(sum(gt_true_list) / len(gt_true_list), input.size(0))
            gt_falses.update(sum(gt_false_list) / len(gt_false_list), input.size(0))
            pred_trues.update(sum(pred_true_list) / len(pred_true_list), input.size(0))
            pred_falses.update(sum(pred_false_list) / len(pred_false_list), input.size(0))
            pred_trues_first.update(sum(pred_true_first_list) / len(pred_true_first_list), input.size(0))

            # record loss
            total_losses.update(total_loss.item(), input.size(0))
            heatmap_losses.update(heatmap_loss.item(), input.size(0))
            # mask_losses.update(mask_loss.item(), input.size(0))
            label_losses.update(label_loss.item(), input.size(0))

            # record metric
            heatmap_ioues.update(sum(heatmap_iou_list) / len(heatmap_iou_list), input.size(0))
            mask_ioues.update(sum(mask_iou_list) / len(mask_iou_list), input.size(0))
            label_acces.update(sum(label_acc_list) / len(label_acc_list), input.size(0))

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
                        loss=total_losses.avg
                        )
            bar.next()
        bar.finish()

    # print(heatmap_losses.avg, heatmap_ioues.avg, \
    #     mask_losses.avg, mask_ioues.avg, \
    #     label_losses.avg, label_acces.avg, \
    #     final_acces.avg)
    return heatmap_losses.avg, heatmap_ioues.avg, \
        mask_ioues.avg, \
        label_losses.avg, label_acces.avg, \
        final_acces.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', metavar='DATASET', default='sad_resnet',
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
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg_resnet_v2',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: hg)')
    parser.add_argument('-s', '--stacks', default=2, type=int, metavar='N',
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
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    # 2 GPU setting  + ResNet-50
    parser.add_argument('--train-batch', default=16, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=16, type=int, metavar='N',
                        help='train batchsize')

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
    parser.add_argument('--snapshot', default=5, type=int,
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
