from __future__ import print_function

import argparse
import os
import random
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F


from imgt_utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


# ############################### Parameters ###############################
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# ******************* Datasets *******************
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_dir', default='imagenet/ILSVRC/Data/CLS-LOC/', type=str, help='imagenet location')

# ******************* Optimization options *******************
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=1024, type=int, metavar='N')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')

# ******************* Checkpoints *******************
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to checkpoint')

# ******************* Architecture *******************
parser.add_argument('--arch', type=str, default='resnet18', help='Teacher Model.')


# ******************* Miscs *******************
parser.add_argument('--manualSeed', type=int, default=123, help='manual seed')

parser.add_argument('--save_dir', default='ckpt_resnet18_kd/', type=str)

# ******************* Device options *******************
parser.add_argument('--gpu-id', default=[0, 1, 2, 3], type=int, nargs='+',
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--apex', default=True, type=bool, help="use APEX or not")

# KD 
parser.add_argument('--arch_t', type=str, default='resnet50', help='Teacher Model.')
parser.add_argument('--teacher_path', type=str, default='', help='Teacher Model location')
parser.add_argument('--temperature', default=20.0, type=float, help='temperature of KD')
parser.add_argument('--alpha', default=0.9, type=float, help='ratio for KL loss')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


os.makedirs(args.save_dir, exist_ok=True)

# ############################### GPU ###############################
use_cuda = torch.cuda.is_available()
device_ids = args.gpu_id
torch.cuda.set_device(device_ids[0])

# ############################### Random seed ###############################


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if args.manualSeed:
    seed_torch(args.manualSeed)


# ##################### APEX #####################

if args.apex:
    try:
        import apex
        USE_APEX = True
        print('Use APEX !!!')
    except ImportError:
        USE_APEX = False
else:
    USE_APEX = False


def loss_fn_kd(outputs, labels, teacher_outputs):
    alpha = args.alpha
    T = args.temperature
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              nn.CrossEntropyLoss()(outputs, labels) * (1. - alpha)
    return KD_loss


def main():
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    # ######################################### Dataset ################################################
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # #################### train / valid dataset ####################
    train_dir = os.path.join(args.img_dir, 'train')
    valid_dir = os.path.join(args.img_dir, 'val')

    trainset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    devset = datasets.ImageFolder(
        valid_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    print('Total images in train, ', len(trainset))
    print('Total images in valid, ', len(devset))

    # #################### data loader ####################
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.workers)

    devloader = data.DataLoader(devset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers, pin_memory=True)

    # ######################################### Model ##################################################
    print("==> creating model: {} ".format(args.arch))
    model = models.__dict__[args.arch]()        # student model
    model.cuda(device_ids[0])

    # teacher model
    print("==> creating teacher model: {} ".format(args.arch_t))
    tch_model = models.__dict__[args.arch_t](pretrained=True)
    tch_model.cuda(device_ids[0])

    # ############################### Optimizer and Loss ###############################
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # ************* USE APEX *************
    if USE_APEX:
        print('Use APEX !!! Initialize Model with APEX')
        model, optimizer = apex.amp.initialize(model, optimizer, loss_scale='dynamic', verbosity=0)

    # ****************** multi-GPU ******************
    model = nn.DataParallel(model, device_ids=device_ids)
    tch_model = nn.DataParallel(tch_model, device_ids=device_ids)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # ############################### Resume ###############################
    title = 'ImageNet'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.save_dir = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.save_dir, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.save_dir, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # load teacher checkpoint
    if args.teacher_path:
        print('Load Teacher model from {} '.format(args.teacher_path))
        checkpoint = torch.load(args.teacher_path, map_location=lambda storage, loc: storage)
        tch_model.load_state_dict(checkpoint['state_dict'])


    print('Evaluate Teacher model .......')
    _, tch_acc = test(devloader, tch_model, criterion, use_cuda)
    print('Teacher Acc. Top-1: {}'.format(tch_acc))

    # ############################### Train and val ###############################
    # save random initialization parameters
    save_checkpoint({'state_dict': model.state_dict()}, False, checkpoint=args.save_dir, filename='init.pth.tar')

    best_acc = -1
    for epoch in range(start_epoch, args.epochs):
        cur_lr = adjust_learning_rate(optimizer, epoch, args)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, cur_lr))

        # train in one epoch
        train_loss, train_acc = train_kd(trainloader, model, tch_model, criterion, optimizer, use_cuda)

        # evaluate
        dev_loss, dev_acc = test(devloader, model, criterion, use_cuda)

        # append logger file
        logger.append([cur_lr, train_loss, dev_loss, train_acc, dev_acc])

        # save model after one epoch
        is_best = dev_acc > best_acc
        best_acc = max(dev_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': dev_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.save_dir, filename='ckpt.pth.tar')

    print('Best val acc:')
    print(best_acc)

    logger.close()


def train_kd(trainloader, model, tch_model, criterion, optimizer, use_cuda):
    # switch to train mode
    model.train()
    tch_model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    print(args)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # teacher model output
        with torch.no_grad():
            outputs_t = tch_model(inputs)

        # compute output
        outputs = model(inputs)

        loss = loss_fn_kd(outputs, targets, outputs_t)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        # ************* USE APEX *************
        if USE_APEX:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        info = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        print(info)
        bar.suffix = info
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def test(testloader, model, criterion, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))



def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    factor = epoch // 30
    if epoch >= 80:
        factor = factor + 1
    lr = args.lr * (0.1 ** factor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr






if __name__ == '__main__':
    main()

