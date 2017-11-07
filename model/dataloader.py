# -*- coding: utf-8 -*-

'''
Dataloader for TIMIT.
'''
from __future__ import print_function
import os
import numpy as np
from tqdm import tqdm
import pdb

glob_X = []  # global container for all feature Xs
glob_Y = []  # global container for all label Ys
spk_dict = dict()  # dictionary {speaker name: speaker id}
dia_dict = dict()  # dictionary {speaker name: dialect id}
gender_dict = {'f': 0, 'm': 1}  # dictionary for male or female
label_count = dict()  # {label_id: count}


def build_info_dict(raw_info_fn):
    '''
    Build dict{spkid: dict{classname: classvalue}}.
    '''
    raw_info = [l.split() for l in open(raw_info_fn)]
    raw_info = raw_info[1:]  # remove header
    info_dict = dict()
    for l in raw_info:
        spkid = l[0]
        birthdate = np.asarray(l[3].split('/'), dtype=np.float64)
        age = (birthdate[0] * 30 + birthdate[1]) / 365. + (86 - birthdate[2])  # age relative to 1986
        h = np.asarray(l[4].rstrip('"').split('\''), dtype=np.float64)
        height = h[0] + h[1] / 12.  # 1 ft = 12 in

        info_dict[spkid] = {'age': age, 'height': height}
    return info_dict


def dataloader(featdir, trainlist, testlist, timitinfo, task):
    info_dict = build_info_dict(timitinfo)
    trnls = [l.rstrip('\n') for l in open(trainlist)]
    tesls = [l.rstrip('\n') for l in open(testlist)]

    # Train
    train_feat = []
    train_label = []
    for l in tqdm(trnls, desc='load train', leave=True):
        spkid = l.split('/')[2][1:]
        train_label.append(info_dict[spkid][task])
        train_feat.append(np.loadtxt(os.path.join(featdir, l), delimiter=';', skiprows=1, usecols=range(1, 6374 + 1)))  # 6374 dim

    # Test
    test_feat = []
    test_label = []
    for l in tqdm(tesls, desc='load test', leave=True):
        spkid = l.split('/')[2][1:]
        test_label.append(info_dict[spkid][task])
        test_feat.append(np.loadtxt(os.path.join(featdir, l), delimiter=';', skiprows=1, usecols=range(1, 6374 + 1)))

    train_feat = np.asarray(train_feat, dtype=np.float64)
    train_label = np.asarray(train_label, dtype=np.float64)
    test_feat = np.asarray(test_feat, dtype=np.float64)
    test_label = np.asarray(test_label, dtype=np.float64)
    return train_feat, train_label, test_feat, test_label
