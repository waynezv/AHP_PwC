# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import errno
import random
import time
import numpy as np

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tensorboard_logger import configure, log_value
from tqdm import tqdm
from colorama import Fore
import pdb

from modelbase.dataloader import dataloader
from modelbase.args import parser
from modelbase.model import Autoencoder, weights_init

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
rng = np.random.RandomState(seed=args.manualSeed)

# CUDA, CUDNN
if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# cudnn.benchmark = True
cudnn.fastest = True

# Init model
print("=> creating model")
netAE = Autoencoder()
# netAE = torch.nn.DataParallel(netAE, device_ids=[1, 2, 3]).cuda()
netAE = netAE.cuda()
netAE.apply(weights_init)
print(netAE)
netAE.apply(weights_init)

# Setup optimizer
optimizerAE = optim.Adam(netAE.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# Prepare data
featdir = '../timit_opensmile_feat'
trainlist = './timit_train_featlist.ctl'
testlist = './timit_test_featlist.ctl'
timitinfo = './timit.info'
task = args.task
print('=> loading data for task ' + Fore.GREEN + '{}'.format(task) + Fore.RESET)
loader_args = {'batch': True, 'batch_size': args.batchSize, 'shuffle': True, 'num_workers': 32}
train_loader, test_loader = dataloader(featdir, trainlist, testlist, timitinfo, task, loader_args)

# Training settings
save_freqency = args.saveFreq  # save every # epoch
old_record_fn = 'youll_never_find_me'  # old record filename
best_test_error = 1e19
best_epoch = 0
# loss_r_meter = ScoreMeter()
# test_err_meter = ScoreMeter()

# Train model
print("=> traning")
for epoch in range(args.nepoch):
    i = 0
    for x, y in train_loader:
        i += 1
        netAE.zero_grad()
        x = Variable(x.cuda())
        # y = Variable(y.cuda())
        xr = netAE(x)
        loss_r = torch.nn.functional.mse_loss(x, xr)  # reconstruction loss
        loss_r.backward()
        optimizerAE.step()

        print('[{:d}/{:d}][{:d}/{:d}] '.format(epoch, args.nepoch, i, len(train_loader)) +
              'LossR: {:.4f} '.format(loss_r.data[0]))
        # Fore.RED + 'LossR: {:.4f} '.format(loss_r.data[0]) + Fore.RESET)
        # Fore.GREEN + 'LossG: {:.4f}'.format(lossG.data[0]) + Fore.RESET)

    # if (i % save_freqency) == 0:
    # Test
    test_err = 0
    for x, y in test_loader:
        x = Variable(x.cuda(), volatile=True)
        xr = netAE(x)
        loss_r = torch.nn.functional.mse_loss(x, xr)
        test_err += loss_r.data[0]
    print(Fore.RED +'Test err: {:.4f}'.format(test_err / float(len(test_loader))) + Fore.RESET)

            # trues = []  # true labels
            # preds = []  # predicted labels
            # # Compute accuracy
            # trues = np.array(list(itertools.chain.from_iterable(trues))).reshape((-1,))
            # preds = np.array(list(itertools.chain.from_iterable(preds))).reshape((-1,))
            # pred_error = 1. - np.count_nonzero(trues == preds) / float(len(trues))
            # print('Trues: ', trues)
            # print('Preds: ', preds)
            # print('Test error: {:.4f}'.format(pred_error))
            # print('Test time: {:.4f}s'.format(time.time() - end_timer))

            # Save best
            # is_best = pred_error < best_test_error
            # if is_best:
                # best_test_error = pred_error
                # best_epoch = epoch
                # best_generation = gen_iterations
                # save_checkpoint({
                    # 'args': args,
                    # 'epoch': epoch,
                    # 'best_epoch': best_epoch,
                    # 'gen_iterations': gen_iterations,
                    # 'best_test_error': best_test_error,
                    # 'netE_state_dict': netE.state_dict(),
                    # 'netP_state_dict': netP.state_dict()
                # }, os.path.join(args.outf, 'checkpoints'), 'checkpoint_BEST.pth.tar')
                # print(Fore.GREEN + 'Saved checkpoint for best test error {:.4f} at epoch {:d}'.format(best_test_error, best_epoch) + Fore.RESET)

            # Logging
            # log_value('test_err', pred_error, gen_iterations)
            # log_value('lossD', lossD.data[0], gen_iterations)
            # test_err_meter.update(pred_error)
            # lossD_meter.update(lossD.data[0])

            # # Checkpointing
            # save_checkpoint({
                # 'args': args,
                # 'epoch': epoch,
                # 'gen_iterations': gen_iterations,
                # 'netE_state_dict': netE.state_dict(),
                # 'netG_state_dict': netG.state_dict(),
                # 'netD_state_dict': netD.state_dict(),
                # 'netP_state_dict': netP.state_dict()
            # }, os.path.join(args.outf, 'checkpoints'), 'checkpoint_gen_{:d}_epoch_{:d}.pth.tar'.format(gen_iterations, epoch))

            # # Delete old checkpoint to save space
            # new_record_fn = os.path.join(args.outf, 'checkpoints', 'checkpoint_gen_{:d}_epoch_{:d}.pth.tar'.format(gen_iterations, epoch))
            # if os.path.exists(old_record_fn) and os.path.exists(new_record_fn):
                # os.remove(old_record_fn)
            # old_record_fn = new_record_fn

# test_err_meter.save('test_err', os.path.join(args.outf, 'records'), 'test_err.tsv')
# lossD_meter.save('lossD', os.path.join(args.outf, 'records'), 'lossD.tsv')
