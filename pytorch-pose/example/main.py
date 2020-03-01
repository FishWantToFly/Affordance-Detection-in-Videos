'''
# training 
python main.py
# resume training from checkpoint
python main.py --resume ./checkpoint/checkpoint_20.pth.tar
# draw line chart
python main.py --resume ./checkpoint/checkpoint_20.pth.tar -w
# visualization
python main.py --resume ./checkpoint/checkpoint_20.pth.tar -e -d 

python main.py --resume ./checkpoint/checkpoint_best_iou.pth.tar -w

python main.py --resume ./checkpoint_0301_iou/checkpoint_best_iou.pth.tar -e
'''

from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

import _init_paths
from pose import Bar
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import accuracy, AverageMeter, final_preds, intersectionOverUnion
from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap, sample_test
from pose.utils.transforms import fliplr, flip_back
import pose.models as models
import pose.datasets as datasets
import pose.losses as losses

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

# get model names and dataset names
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


dataset_names = sorted(name for name in datasets.__dict__
    if name.islower() and not name.startswith("__")
    and callable(datasets.__dict__[name]))


# init global variables
best_acc = 0
best_iou = 0
idx = []

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

    # plt.xlabel('Epoch')
    # plt.plot(epoch_idx_list, train_acc_list)
    # plt.plot(epoch_idx_list, val_acc_list)
    # plt.legend(['Train acc', 'Val acc'], loc='upper left')
    # plt.savefig(os.path.join(args.checkpoint, 'log_acc.png'))
    # plt.cla()

    plt.xlabel('Epoch')
    plt.plot(epoch_idx_list, val_iou_list)
    plt.legend(['Val iou'], loc='upper left')
    plt.savefig(os.path.join(args.checkpoint, 'log_iou.png'))
    plt.cla()

def main(args):
    global best_acc
    global best_iou
    global idx

    # idx is the index of joints used to compute accuracy
    if args.dataset in ['mpii', 'lsp']:
        idx = [1,2,3,4,5,6,11,12,15,16]
    elif args.dataset == 'coco':
        idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    elif args.dataset == 'sad':
        idx = [1] # support affordance
    else:
        print("Unknown dataset: {}".format(args.dataset))
        assert False

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
    criterion = losses.IoULoss().to(device)
    # criterion = losses.JointsMSELoss().to(device)

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
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Val IoU'])

    print('    Total params: %.2fM'
          % (sum(p.numel() for p in model.parameters())/1000000.0))

    '''
    datasets.__dict__[args.dataset] -> depend on args.dataset to replace with datasets
    '''
    # create data loader
    train_dataset = datasets.__dict__[args.dataset](is_train=True, **vars(args))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )

    # for i, (img, target, meta) in enumerate(train_loader):
    #     if i == 10 : return
    #     print("target shape :")
    #     print(target.shape)

    val_dataset = datasets.__dict__[args.dataset](is_train=False, **vars(args))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # write line-chart
    if args.write: 
        draw_line_chart(args, os.path.join(args.checkpoint, 'log.txt'))
        return

    # evaluation only
    global JUST_EVALUATE
    JUST_EVALUATE = False
    if args.evaluate:
        print('\nEvaluation only')
        JUST_EVALUATE = True
        loss, iou, predictions = validate(val_loader, model, criterion, njoints,
                                           args.checkpoint, args.debug, args.flip)
        return

    # train and eval
    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        # for test 10 epoch
        if args.test and epoch == 11:
            break

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
        }, predictions, is_best_iou, checkpoint=args.checkpoint, snapshot=args.snapshot)

    logger.close()

    print("Best iou = %.3f" % (best_iou))
    draw_line_chart(args, os.path.join(args.checkpoint, 'log.txt'))

def train(train_loader, model, criterion, optimizer, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    gt_win, pred_win = None, None
    bar = Bar('Train', max=len(train_loader))
    for i, (input, target, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device, non_blocking=True)
        target_weight = meta['target_weight'].to(device, non_blocking=True)

        # compute output
        output = model(input)

        if type(output) == list:  # multiple output # beacuse of intermediate prediction
            loss = 0
            for o in output:
                loss += criterion(o, target, target_weight)
            output = output[-1]
        else:  # single output
            loss = criterion(output, target, target_weight)
        acc = accuracy(output, target, idx)
        
        # measure accuracy and record loss
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
        for i, (input, target, meta) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target_weight = meta['target_weight'].to(device, non_blocking=True)

            # compute output
            output = model(input)
            score_map = output[-1].cpu() if type(output) == list else output.cpu()
            if flip:
                flip_input = torch.from_numpy(fliplr(input.clone().numpy())).float().to(device)
                flip_output = model(flip_input)
                flip_output = flip_output[-1].cpu() if type(flip_output) == list else flip_output.cpu()
                flip_output = flip_back(flip_output)
                score_map += flip_output



            if type(output) == list:  # multiple output
                loss = 0
                for o in output:
                    loss += criterion(o, target, target_weight)
                output = output[-1]
            else:  # single output
                loss = criterion(output, target, target_weight)

            # if i == 10: break

            # acc = accuracy(score_map, target.cpu(), idx)
            iou = intersectionOverUnion(output.cpu(), target.cpu(), idx) # have not tested

            # generate predictions
            # preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # for n in range(score_map.size(0)):
            #     predictions[meta['index'][n], :, :] = preds[n, :, :]

            if debug:
                gt_batch_img = batch_with_heatmap(input, target, 'gt')
                pred_batch_img = batch_with_heatmap(input, score_map, 'pred')
                if not gt_win or not pred_win:
                    ax1 = plt.subplot(121)
                    ax1.title.set_text('Groundtruth')
                    gt_win = plt.imshow(gt_batch_img)
                    ax2 = plt.subplot(122)
                    ax2.title.set_text('Prediction')
                    pred_win = plt.imshow(pred_batch_img)
                    
                else:
                    gt_win.set_data(gt_batch_img)
                    pred_win.set_data(pred_batch_img)
                #### visualize by pop out window 
                # plt.pause(.5)
                # plt.draw() 
                ### save in fig
                save_fig_dir = os.path.join(checkpoint, 'vis')
                if not isdir(save_fig_dir):
                    mkdir_p(save_fig_dir)
                plt.plot()
                plt.savefig(os.path.join(save_fig_dir, '%d.png' % (i)))

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            # acces.update(acc[0], input.size(0))
            ioues.update(iou, input.size(0))

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
    
    print("IoU: ")
    print("%.3f" % (ioues.avg))
    return losses.avg, ioues.avg, predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Dataset setting
    # default is mpii at first
    parser.add_argument('--dataset', metavar='DATASET', default='sad',
                        choices=dataset_names,
                        help='Datasets: ' +
                            ' | '.join(dataset_names) +
                            ' (default: mpii)')
    parser.add_argument('--image-path', default='/home/s5078345/Affordance-Detection-on-Video/dataset', type=str,
                        help='path to images')
    parser.add_argument('--anno-path', default='', type=str,
                        help='path to annotation (json)')
    parser.add_argument('--year', default=2014, type=int, metavar='N',
                        help='year of coco dataset: 2014 (default) | 2017)')

    # original was 256 and 64 -> 480 adn 120 ???

    parser.add_argument('--inp-res', default=256, type=int,
                        help='input resolution (default: 256)')
    parser.add_argument('--out-res', default=64, type=int,
                    help='output resolution (default: 64, to gen GT)')
                    
    parser.add_argument('--dataset-list-dir-path', default='/home/s5078345/Affordance-Detection-on-Video/dataset/data_list', type=str,
                    help='dir of train/test data list')



    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: hg)')
    parser.add_argument('-s', '--stacks', default=8, type=int, metavar='N',
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
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    # 2 GPU setting
    # parser.add_argument('--train-batch', default=20, type=int, metavar='N', # if andy takes GPU
    #                     help='train batchsize')
    parser.add_argument('--train-batch', default=30, type=int, metavar='N', # IoU loss
                        help='train batchsize')
    # parser.add_argument('--train-batch', default=30, type=int, metavar='N', # normal
                        # help='train batchsize')
    # parser.add_argument('--test-batch', default=1, type=int, metavar='N', # for debug
    #                     help='test batchsize')
    parser.add_argument('--test-batch', default=10, type=int, metavar='N',
                        help='test batchsize')

    # 2020.2.24 2.5e-4 -> 2.0e-4
    parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
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
    # parser.add_argument('--scale-factor', type=float, default=0.25,
    #                     help='Scale factor (data aug).')
    # parser.add_argument('--rot-factor', type=float, default=30,
    #                     help='Rotation factor (data aug).')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    # parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
    #                     choices=['Gaussian', 'Cauchy'],
    #                     help='Labelmap dist type: (default=Gaussian)')
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




    main(parser.parse_args())
