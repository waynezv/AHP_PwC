from __future__ import print_function
import os
import sys
import time
import shutil
import numpy as np
import torch
from colorama import Fore


def create_save_folder(save_path, force=False, ignore_patterns=[]):
    '''
    Create new folder and backup old folder.
    '''
    if os.path.exists(save_path):
        print(Fore.RED + save_path + Fore.RESET +
              ' already exists!', file=sys.stderr)
        if not force:
            ans = input('Do you want to overwrite it? [y/N]:')
            if ans not in ('y', 'Y', 'yes', 'Yes'):
                os.exit(1)
        from getpass import getuser
        tmp_path = '/tmp/{}-experiments/{}_{}'.format(getuser(),
                                                      os.path.basename(save_path),
                                                      time.time())
        print('move existing {} to {}'.format(save_path, Fore.RED +
                                              tmp_path + Fore.RESET))
        shutil.copytree(save_path, tmp_path)
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    print('create folder: ' + Fore.GREEN + save_path + Fore.RESET)

    # copy code to save folder
    if save_path.find('debug') < 0:
        shutil.copytree('.', os.path.join(save_path, 'src'), symlinks=True,
                        ignore=shutil.ignore_patterns('*.pyc', '__pycache__',
                                                      '*.path.tar', '*.pth',
                                                      '*.ipynb', '.*', 'data',
                                                      'save', 'save_backup',
                                                      save_path,
                                                      *ignore_patterns))


def save_checkpoint(state, save_dir, filename, is_best=False):
    '''
    Save training checkpoints.
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))


class ScoreMeter():
    '''
    Record, measure, save scores.
    '''
    def __init__(self):
        self.score = []
        self.subscore = []
        self.avg = 0

    def update(self, val):
        self.subscore.append(val)

    def reset(self):
        self.avg = np.mean(self.subscore)
        self.score.append(self.avg)
        self.subscore = []

    def save(self, score_name, save_dir, fn):
        scores = "idx\t{}".format(score_name)
        for i, s in enumerate(self.score):
            scores += "\n"
            scores += "{:d}\t{:.4f}".format(i, s)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fn = os.path.join(save_dir, fn)
        with open(fn, 'w') as f:
            f.write(scores)


def adjust_learning_rate(optimizer, lr_init, decay_rate, epoch, num_epochs):
    '''
    Decay Learning rate at 1/2 and 3/4 of the num_epochs.
    '''
    lr = lr_init
    if epoch >= num_epochs * 0.75:
        lr *= decay_rate**2
    elif epoch >= num_epochs * 0.5:
        lr *= decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
