s'''
2020.3.6
only for test 
1. feature size from model
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

RELABEL = False

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # There is BN issue for early version of PyTorch
                        # see https://github.com/bearpaw/pytorch-pose/issues/33


def main(args):
    global best_acc
    global best_iou
    global idx

    # 2020.3.2
    global REDRAW

    # 2020.3.4
    # if you do type arg.resume
    # args.checkpoint would be derived from arg.resume
    args.train_batch = 6
    args.test_batch = 6

    if args.dataset == 'sad':
        idx = [1] # support affordance


    # create model
    njoints = datasets.__dict__[args.dataset].njoints

    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    model = models.__dict__[args.arch](num_stacks=args.stacks,
                                       num_blocks=args.blocks,
                                       num_classes=njoints,
                                       resnet_layers=args.resnet_layers)


    model = torch.nn.DataParallel(model).to(device)
    criterion = losses.IoULoss().to(device)
    criterion_semantic = losses.SemanticLoss().to(device)
    criterions = [criterion, criterion_semantic]


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



    print('\nFor test only')
    loss, iou, predictions = validate(val_loader, model, criterions, njoints,
                                        args.checkpoint, args.debug, args.flip)




'''
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
        target_weight = meta['target_weight'].to(device, non_blocking=True)

        # compute output
        # output = model(input)
        output = model(torch.cat((input, input_depth), 1))

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
'''

def validate(val_loader, model, criterions, num_classes, checkpoint, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ioues = AverageMeter()

    criterion, criterion_semantic = criterions

    # predictions
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)

    # switch to evaluate mode
    model.eval()

    gt_win, pred_win = None, None
    iou = None
    end = time.time()
    bar = Bar('Eval ', max=len(val_loader))
    with torch.no_grad():
        for i, (input, input_depth, target, target_semantic, meta) in enumerate(val_loader):
            if i == 5: break

            # measure data loading time
            data_time.update(time.time() - end)

            input = input.to(device, non_blocking=True)
            input_depth = input_depth.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target_weight = meta['target_weight'].to(device, non_blocking=True)

            target_semantic = target_semantic.to(device, non_blocking=True)

            # compute output
            # output = model(input)
            output, out_semantic = model(torch.cat((input, input_depth), 1))
            score_map = output[-1].cpu() if type(output) == list else output.cpu()
            # print()
            # print(len(out_test))
            # print(out_test[0].shape)
            # print(target_semantic)
            # print(target_semantic.shape)

            
            if type(output) == list: # multiple output
                loss = 0
                for o in output:
                    loss += criterion(o, target, target_weight)
                for o_sem in out_semantic:
                    pass
                    loss += criterion_semantic(o_sem, target_semantic)
                    # print(loss)
                output = output[-1]
                output_semantic = out_semantic[-1]
            else:  # single output
                loss = criterion(output, target, target_weight)


            ## measure semantic accuracy
            # print(target_semantic.shape)
            # print(output_semantic.shape)
            _, semantic_predict = torch.max(output_semantic, 1)

            print(semantic_predict)
            print(target_semantic)
            print((semantic_predict == target_semantic).sum())


            '''
            iou = intersectionOverUnion(output.cpu(), target.cpu(), idx) # have not tested

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
            '''
        bar.finish()
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
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    # 2 GPU setting
    # parser.add_argument('--train-batch', default=20, type=int, metavar='N', # if andy takes GPU
    #                     help='train batchsize')
    parser.add_argument('--train-batch', default=30, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=10, type=int, metavar='N',
                        help='test batchsize')

    # parser.add_argument('--train-batch', default=1, type=int, metavar='N',
    #                     help='train batchsize')
    # parser.add_argument('--test-batch', default=1, type=int, metavar='N', # for debug
    #                     help='test batchsize')

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

    main(parser.parse_args())
