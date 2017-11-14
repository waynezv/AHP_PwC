# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import errno
import random
import numpy as np
from colorama import Fore
import pdb

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn

from modelbase.dataloader import dataloader
from modelbase.args import parser
from modelbase.model import Regressor, weights_init
from modelbase.utils import save_checkpoint

# Parse args
args = parser.parse_args()

# Make dirs
try:
    os.makedirs(args.outf)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# Fix seed for randomization
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

# CUDA, CUDNN
if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# cudnn.benchmark = True
cudnn.fastest = True

# Init model
if args.resume:  # Resume from saved checkpoint
    if os.path.isfile(args.resume):
        print('=> loading checkpoint {}'.format(Fore.YELLOW + args.resume + Fore.RESET))
        checkpoint = torch.load(args.resume)

        print("=> creating model")
        netR = Regressor().cuda()

        netR.load_state_dict(checkpoint['netR_state_dict'])
        print("=> loaded model with checkpoint '{}' (epoch {})".
              format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".
              format(Fore.RED + args.resume + Fore.RESET), file=sys.stderr)
        sys.exit(0)

else:
    print("=> creating model")
    netR = Regressor().cuda()
    print(netR)
    netR.apply(weights_init)

# Prepare data
task = args.task
featdir = '../timit_opensmile_feat'
trainlist = './timit_train_featlist.ctl'
testlist = './timit_test_featlist.ctl'
timitinfo = './timit.info'
print('=> loading data for task ' + Fore.GREEN + '{}'.format(task) + Fore.RESET)
loader_args = {'batch': True, 'batch_size': args.batchSize, 'shuffle': True, 'num_workers': 32}
train_loader, test_loader = dataloader(featdir, trainlist, testlist, timitinfo, task, loader_args)

# Eval
if args.eval:
    print("=> evaluating model")
    if not os.path.exists(os.path.join(args.outf, 'eval')):
        os.makedirs(os.path.join(args.outf, 'eval'))

    # test_feat = np.loadtxt(args.testFn, delimiter=';', skiprows=1, usecols=range(1, 6374 + 1))
    # x = torch.from_numpy(test_feat).float().view(1, -1)
    # x = Variable(x.cuda(), volatile=True)
    # yp = netR(x)
    # print("Predicted {} for '{}': {:.4f}".format(args.task, args.testFn, yp.data[0]))

    test_err = 0  # average test error
    for x, y in test_loader:
        x = Variable(x.cuda(), volatile=True)
        y = Variable(y.float().cuda())
        yp = netR(x)
        loss_mae = (yp - y).abs().mean()
        test_err += loss_mae.data[0]
    test_err = test_err / float(len(test_loader))
    print(Fore.RED + 'Mean absolute error: {:.4f}'.format(test_err) + Fore.RESET)
    sys.exit(0)

# Setup optimizer
optimizerR = optim.Adam(netR.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# Training settings
old_record_fn = 'youll_never_find_me'  # old record filename
best_test_err = 1e19
best_epoch = 0

# Train model
print("=> traning")
for epoch in range(args.nepoch):
    i = 0
    for x, y in train_loader:
        i += 1
        netR.zero_grad()
        x = Variable(x.cuda())
        y = Variable(y.float().cuda())
        yp = netR(x)
        loss_mae = (yp - y).abs().mean()  # mean absolute error
        loss_mae.backward()
        optimizerR.step()
        print('[{:d}/{:d}][{:d}/{:d}] '.format(epoch, args.nepoch, i, len(train_loader)) +
              'loss_mae: {:.4f}'.format(loss_mae.data[0]))

    # Test
    test_err = 0  # average test error
    for x, y in test_loader:
        x = Variable(x.cuda(), volatile=True)
        y = Variable(y.float().cuda())
        yp = netR(x)
        loss_mae = (yp - y).abs().mean()
        test_err += loss_mae.data[0]
    test_err = test_err / float(len(test_loader))
    print(Fore.RED + 'Test error: {:.4f}'.format(test_err) + Fore.RESET)

    # Save best
    if not os.path.exists(os.path.join(args.outf, 'checkpoints')):
        os.makedirs(os.path.join(args.outf, 'checkpoints'))
    is_best = test_err < best_test_err
    if is_best:
        best_test_err = test_err
        best_epoch = epoch
        save_checkpoint({
            'args': args,
            'epoch': epoch,
            'best_epoch': best_epoch,
            'best_test_err': best_test_err,
            'netR_state_dict': netR.state_dict()
        }, os.path.join(args.outf, 'checkpoints'), 'checkpoint_BEST_{}.pth.tar'.format(args.task))
        print(Fore.GREEN + 'Saved checkpoint for best test error {:.4f} at epoch {:d}'.
              format(best_test_err, best_epoch) + Fore.RESET)

    # Checkpointing
    save_checkpoint({
        'args': args,
        'epoch': epoch,
        'netR_state_dict': netR.state_dict(),
    }, os.path.join(args.outf, 'checkpoints'), 'checkpoint_epoch_{:d}.pth.tar'.format(epoch))

    # Delete old checkpoint to save space
    new_record_fn = os.path.join(args.outf, 'checkpoints', 'checkpoint_epoch_{:d}.pth.tar'.format(epoch))
    if os.path.exists(old_record_fn) and os.path.exists(new_record_fn):
        os.remove(old_record_fn)
    old_record_fn = new_record_fn
