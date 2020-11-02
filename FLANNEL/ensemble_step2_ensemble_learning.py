from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import torchvision.models as models
import explore_version_03.models.proposedModels.models as models 
from explore_version_03.data.ensemble_dataset import EnsembleDataset
from explore_version_03.data.ensemble_dataset import EnsembleDatasetSampling
from explore_version_03.utils import Bar, AverageMeter, accuracy, mkdir_p
from explore_version_03.utils.logger import Logger, savefig
import csv
from explore_version_03.utils.measure import MeasureR
from explore_version_03.models.proposedModels.loss import FocalLoss as focalloss
# cv3: 0.001

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

#customized_models_names = sorted(name for name in customized_models.__dict__
#    if name.islower() and not name.startswith("__")
#    and callable(customized_models.__dict__[name]))
#
#for name in customized_models.__dict__:
#    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
#        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names
#+ customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Experiment ID
parser.add_argument('--experimentID', default='%s_20200719_gamma_10_multiclass_cv5_focal', type=str, metavar='E_ID',
                    help='ID of Current experiment')
parser.add_argument('--cv', default='cv5', type=str, metavar='E_ID',
                    help='ID of Current experiment')
parser.add_argument('--data_dir', default='./explore_version_03/results_class_ensemble/%s_20200407_multiclass_%s', type=str, metavar='E_ID',
                    help='ID of Current experiment')

# Datasets
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--load_size', type=int, default=336, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=224, help='then crop to this size')
parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                    help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser.add_argument('-d', '--data', default='./data_preprocess/standard_data_multiclass_0325/exp_%s_list.pkl', type=str)

# Optimization options
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=20, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=20, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--serial_batches', action='store_true',
                    help='if true, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0.5, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[50, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.5, type=float, metavar='M',
                    help='momentum')      
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='./explore_version_03/checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('-ck_n', '--checkpoint_saved_n', default=2, type=int, metavar='saved_N',
                    help='each N epoch to save model')

# Test Outputs
parser.add_argument('--test', default = True, dest='test', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('--results', default='./explore_version_03/results', type=str, metavar='PATH',
                    help='path to save experiment results (default: results)')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='saved model ID for loading checkpoint (default: none)')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='ensembleNovel',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--pretrained', dest='pretrained', default=True, action='store_false',
                    help='use pre-trained model')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    
    experimentID = args.experimentID%args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    
    if not os.path.isdir(os.path.join(args.checkpoint, experimentID)):
        mkdir_p(os.path.join(args.checkpoint, experimentID))
    
    checkpoint_dir = os.path.join(args.checkpoint, experimentID)
    
    # Data loading code
    train_dataset = EnsembleDataset(args, 'train')
    train_distri = train_dataset.get_label_distri()
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_batch,
                                               shuffle=not args.serial_batches,
                                               num_workers=int(args.workers))

    valid_dataset = EnsembleDataset(args, 'valid')
    val_loader = torch.utils.data.DataLoader(valid_dataset,
                                             batch_size=args.test_batch,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    test_dataset = EnsembleDataset(args, 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.test_batch,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    print (train_distri)
#    return
    criterion = focalloss(gamma=10, label_distri = train_distri, model_name = args.arch, cuda_a = use_cuda)
#    criterion = nn.CrossEntropyLoss()
#    criterion = nn.KLDivLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.test is False:
      # Resume
      title = args.arch
      if args.resume:
          # Load checkpoint.
          print('==> Resuming from checkpoint..')
          checkpoint_path = os.path.join(checkpoint_dir,args.resume+'.checkpoint.pth.tar')
          print (checkpoint_path)
          assert os.path.isfile(checkpoint_path), 'Error: no checkpoint directory found!'
          checkpoint = torch.load(checkpoint_path)
          best_acc = checkpoint['best_acc']
          start_epoch = checkpoint['epoch']
          model.load_state_dict(checkpoint['state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer'])
          logger = Logger(os.path.join(checkpoint_dir, 'log.txt'), title=title, resume=True)
      else:
          logger = Logger(os.path.join(checkpoint_dir, 'log.txt'), title=title)
          logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.test:
        print('\Test only')
        if len(args.resume) > 0:
          print ('load %s-th checkpoint'%args.resume)
          checkpoint_path = os.path.join(checkpoint_dir,args.resume+'.checkpoint.pth.tar')
        else:
          print ('load best checkpoint')
          checkpoint_path = os.path.join(checkpoint_dir,'model_best.pth.tar')
        print (checkpoint_path)
        assert os.path.isfile(checkpoint_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(checkpoint_path)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
          
        if not os.path.isdir(args.results):
            mkdir_p(args.results)
        if not os.path.isdir(os.path.join(args.results, experimentID)):
            mkdir_p(os.path.join(args.results, experimentID))
        results_dir = os.path.join(args.results, experimentID)
        test_loss, test_acc, pred_d, real_d = test(test_loader, model, criterion, start_epoch, use_cuda)
        
        with open(os.path.join(results_dir, 'result_detail.csv'), 'w') as f:
            csv_writer = csv.writer(f)
            for i in range(len(real_d)):
                x = np.zeros(len(pred_d[i]))
                x[real_d[i]] = 1
                y = np.exp(pred_d[i])/np.sum(np.exp(pred_d[i]))
                csv_writer.writerow(list(y) + list(x))

        mr = MeasureR(results_dir, test_loss, test_acc)
        mr.output()
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return
    
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc, _, _ = test(val_loader, model, criterion, epoch, use_cuda)
        l_loss, l_acc, _, _ = test(test_loader, model, criterion, epoch, use_cuda)
        
        print (train_loss, train_acc, test_acc, l_acc)
        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if epoch%args.checkpoint_saved_n == 0:
          save_checkpoint({
                  'epoch': epoch,
                  'state_dict': model.state_dict(),
                  'acc': test_acc,
                  'best_acc': best_acc,
                  'optimizer' : optimizer.state_dict(),
              }, epoch, is_best, checkpoint=checkpoint_dir)

    logger.close()
    logger.plot()
    savefig(os.path.join(checkpoint_dir, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, databatch in enumerate(train_loader):
        inputs = databatch['A']
        targets = databatch['B']
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs).float(), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        if use_cuda:
          loss = criterion(outputs, targets.type(torch.LongTensor).cuda())
        else:
          loss = criterion(outputs, targets.type(torch.LongTensor))
        
        # print ('train', loss)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 1))
        # print (loss.data, prec1, prec5)
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    real_labels = []
    pred_labels = []
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, databatch in enumerate(val_loader):
        inputs = databatch['A']
        targets = databatch['B']
        # measure data loading time
        data_time.update(time.time() - end)
        real_labels += list(targets)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets= torch.autograd.Variable(inputs, volatile=True).float(), torch.autograd.Variable(targets)
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # print('test', loss)
        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data, topk=(1, 1))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        pred_labels.append(outputs.detach().cpu().numpy())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
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
    return (losses.avg, top1.avg, np.concatenate(pred_labels, 0), np.array(real_labels))

def save_checkpoint(state, epoch_id, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, str(epoch_id)+'.'+filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()

