# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser()

# Task settings
parser.add_argument('--task', default='age', help='task to run: age, height, lie. default=age')

# Train settings
parser.add_argument('--lr', type=float, default=0.001, help='learning rate. default=0.001')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size. default=64')
parser.add_argument('--nepoch', type=int, default=30, help='number of epochs to train for. default=30')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--outf', default='', help='folder to save results and model checkpoints')
parser.add_argument('--saveFreq', type=int, default=100, help='number of updates to save checkpoints after. default=100')

# Resume settings
parser.add_argument('--resume', default='', help='path of saved checkpoint to resume from')
parser.add_argument('--eval', action='store_true', help='evaluate trained model')
parser.add_argument('--testFn', default='', help='test filename')
