# -*- coding: utf-8 -*-

import math
import torch.nn as nn
import pdb


class Autoencoder(nn.Module):
    '''
    Autoencoder.
    '''
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6374, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 20)
        )
        self.decoder = nn.Sequential(
            nn.Linear(20, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 6374)
        )
        self.predictor = nn.Sequential(
            nn.Linear(20, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z), self.predictor(z).view(-1)


class Regressor(nn.Module):
    '''
    Regressor.
    '''
    def __init__(self):
        super(Regressor, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(6374, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.LeakyReLU(0.2),
            nn.Linear(40, 1)
        )

    def forward(self, x):
        return self.regressor(x).view(-1)


def weights_init(m):
    '''
    Custom weights initialization.
    '''
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight.data, a=0.2, mode='fan_in')
        m.bias.data.zero_()
