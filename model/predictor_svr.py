# -*- coding: utf-8 -*-

'''
Support vector regression for age, height.
'''

from __future__ import print_function
import numpy as np
from sklearn.svm import SVR
from modelbase.dataloader import dataloader
import pdb

# Prepare data
featdir = '../timit_opensmile_feat'
trainlist = './timit_train_featlist.ctl'
testlist = './timit_test_featlist.ctl'
timitinfo = './timit.info'
task = 'gender'

Xtrn, Ytrn, Xtes, Ytes = dataloader(featdir, trainlist, testlist, timitinfo, task)

# Train
svr = SVR(C=1, epsilon=0.2, shrinking=True,
          kernel='rbf', degree=3, gamma='auto',
          tol=0.001, verbose=True, max_iter=-1)
svr.fit(Xtrn, Ytrn)

# Test
Ypred = svr.predict(Xtes)
pred_mse = np.abs(Ypred - Ytes).mean()
print('{} MAE = {:.4f}'.format(task, pred_mse))
