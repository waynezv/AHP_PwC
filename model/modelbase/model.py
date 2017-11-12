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
            nn.Linear(6374, 2000),
            nn.BatchNorm1d(2000),
            nn.LeakyReLU(0.2),
            nn.Linear(2000, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.2),
            nn.Linear(500, 200),
            nn.BatchNorm1d(200),
        )
        self.decoder = nn.Sequential(
            nn.Linear(200, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.2),
            nn.Linear(500, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 2000),
            nn.BatchNorm1d(2000),
            nn.LeakyReLU(0.2),
            nn.Linear(2000, 6374),
            nn.BatchNorm1d(6374),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


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
