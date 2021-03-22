#!/usr/bin/env python
import os
import time
import argparse
import random
import warnings
import torch.nn as nn
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torchvision import datasets, models, transforms
from utils import *
from torch.utils.tensorboard import SummaryWriter
from models.se_resnet import *
from models.resnet_cbam import *

parse = argparse.ArgumentParser(description="PyTorch Training")
parse.add_argument('--lr', default=0.1, type=float, help='learning rate')
parse.add_argument('--epoch', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parse.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parse.add_argument('--netname', default='se_resnet20', type=str, help='train network name')
parse.add_argument('-j', '--workers', default=1, type=int, metavar='N',help='number of data loading workers (default: 4)')
parse.add_argument('--baseline', default=False, action='store_true', help='choose origin_net or senet')
parse.add_argument('--resume', '-r', default=False, action='store', help='resume from checkpoint')
parse.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parse.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parse.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parse.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parse.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parse.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parse.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parse.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parse.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parse.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# 硬件设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    args = parse.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')


    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    model = se_resnet20()
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    steps = 16
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    total_epoch = args.start_epoch + args.epoch
    global_step = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            if args.gpu is not None:
                # best_acc may be from a checkpoint from a different GPU
                best_acc = best_acc.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    data_dir = 'data/data_map'
    train_folder = os.path.join(data_dir, "train")
    val_folder = os.path.join(data_dir, "val")


    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_set = datasets.ImageFolder(train_folder, transform_train)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_set = datasets.ImageFolder(val_folder, transform_val)
    valloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    if args.evaluate:
        validate(valloader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, total_epoch):
        for idx in range(steps):
            scheduler.step()
            print(scheduler.get_lr())
        print('Reset scheduler')

        # train for one epoch
        train(trainloader , model, epoch, total_epoch, criterion, optimizer,args)

        # evaluate on validation set
        acc = validate(valloader ,model, criterion, args)

        best_acc = 0
        if acc > best_acc:
            best_acc = acc
            print("Saveing model...")
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'global_step': global_step
            }
            if not os.path.isdir('./checkpoint/slam_map/{}'.format(args.netname)):
                os.mkdir('./checkpoint/slam_map/{}'.format(args.netname))
            torch.save(state, ('./checkpoint/slam_map/{}/Epoch{}_acc{:.2f}_ckpt.pth'.format(args.netname, epoch, acc)))
            torch.save(state, ('./checkpoint/slam_map/{}/best_acc_ckpt.pth'.format(args.netname, epoch, acc)))

# train
def train(trainloader,model, epoch, total_epoch, criterion, optimizer,args):
    global global_step
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.2f')
    top1 = AverageMeter('Acc1', ':6.2f')
    top2 = AverageMeter('Acc2', ':6.2f')
    progress = ProgressMeter(
        len(trainloader),
        [batch_time, data_time, losses, top1, top2, ],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    global_step = 0

    for batch_idx, (inputs, labels) in enumerate(trainloader):

        global_step += 1
        data_time.update(time.time() - end)

        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(args.gpu, non_blocking=True)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        acc1, acc2 = accuracy(outputs, labels, topk=(1,2))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top2.update(acc1[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)

        if batch_idx % 50 == 0:
            print('Epoch: [{0}/{1}] [{2}/{3}]'
                'Loss(avg): {loss.val:.4f}({loss.avg:.4f}), '
                'Top1 acc(avg): {top1.val:.3f}({top1.avg:.3f}), '
                'Top2 acc(avg): {top2.val:.3f}({top2.avg:.3f})'.format(
                epoch,
                total_epoch,
                batch_idx,
                len(trainloader),
                loss=losses,
                top1=top1,
                top2=top2))
            print(' * Top1 avg_acc {top1.avg:.3f} ,Top2 acc(avg): {top2.val:.3f}({top2.avg:.3f})'.format(top1=top1,top2=top2))


# validate
def validate(valloader, model, criterion,args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(
        len(valloader),
        [batch_time, losses, top1, top2],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_idx, (inputs, labels) in enumerate(valloader):

            if args.gpu is not None:
                inputs = inputs.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(args.gpu, non_blocking=True)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # measure accuracy and record loss
            acc1, acc2 = accuracy(outputs, labels, topk=(1,2))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top2.update(acc2[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % args.print_freq == 0:
                progress.display(batch_idx)

            if batch_idx % 50 == 0:
                print(' Epoch: [{0}/{1}]'
                      ' Loss(avg): {loss.val:.4f}({loss.avg:.4f}), '
                      ' Top1 acc(avg): {top1.val:.3f}({top1.avg:.3f}), '
                    ' Top2 acc(avg): {top2.val:.3f}({top2.avg:.3f})'.format(
                    batch_idx, len(valloader), loss=losses, top1=top1, top2=top2))

        return top1.avg



if __name__ == "__main__":
    main()
    #predict()

