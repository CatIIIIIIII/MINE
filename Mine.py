# -*- coding: UTF-8 -*-
"""=================================================
@IDE     :Pycharm
@Author  :Ning Wang
@Contact :@qq 574386703
@Date    :2020.9.22
@Desc    :The module is the implement of mutual information neural estimation, it is used widely
          in measuring the distance between 2 distributions like GAN or self-supervised DL.
          See more details in readme.pdf
=================================================="""
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import math


def log_sum_exp(x, axis=None):
    """
    :param x: pytorch tensor
    :param axis: the dimension to
    :return: y = log(sum(exp(x-x_max)))
    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y


def fGAN(p_samples, q_samples, measure=None):
    """
    :param p_samples: positive samples
    :param q_samples: negative samples
    :param measure: different measurement, support GAN, JSD, X2, KL, H2 now
    :return: expectation of positive samples and negative samples
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)
        Eq = F.softplus(-q_samples) + q_samples - log_2  # Note JSD will be shifted
    elif measure == 'X2':
        Ep = p_samples ** 2
        Eq = 0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Ep = p_samples
        Eq = torch.exp(q_samples - 1.)
    # elif measure == 'RKL':
    #     Ep = -torch.exp(-p_samples)
    #     Eq = q_samples - 1.
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
        Eq = torch.exp(q_samples) - 1.
    else:
        NotImplementedError("Check your measurement! Support GAN, JSD, X2, KL, H2 now")
        Ep, Eq = None, None

    return Ep, Eq


class T_net(nn.Module):
    """
    define neural network to estimate function T
    """

    def __init__(self, num_input, num_hidden):
        """
        :param num_input: the dimension of input layer
        :param num_hidden: the dimension of hidden layer
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_input, num_hidden),
            nn.ELU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ELU(),
            nn.Linear(num_hidden, 1)
        )
        self._init()

    def _init(self):
        """Init network parameters."""
        net = self.model
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=1e-3)

    def forward(self, x, z):
        """Forward propagation"""
        T_input = torch.stack([x, z], dim=-1)
        return self.model(T_input)


class MINE:
    def __init__(self, num_input, num_hidden=128, lr=1e-3, ma_rate=0.01,
                 mode='DV', measure='GAN', device='cpu'):
        """
        :param num_input: input dimension of X, which should equal to dimension of Y
        :param num_hidden: hidden unit dimension of NN
        :param lr: learning rate to update MINE
        :param ma_rate: moving average rate in mode 'DV'
        :param mode: three basics of calculating mutual information:
                    -DV, fGAN, NCE
        :param measure: for different modes, there are different measurement
                    -DV: naive, ma (see more details in https://arxiv.org/pdf/1801.04062.pdf)
                    -fGAN: GAN, JSD, X2, KL, H2(see more details in https://arxiv.org/pdf/1606.00709.pdf)
                    -NCE: info (see more details in https://arxiv.org/pdf/1808.06670.pdf)
        :param device: cuda or cpu

        please note that DV and NCE based measurements need large number of negative samples
        """
        self.device = device
        self.update_mode = mode
        self.measure = measure
        self.ma_rate = ma_rate
        self.T = T_net(num_input=num_input,
                       num_hidden=num_hidden,
                       ).to(self.device)
        self.optimizer = optim.Adam(self.T.parameters(), lr=lr)
        self.et_ma = 1.

    def get_loss(self, samples):
        """
        :param samples: the samples to get loss in 'torch.float32' type. it should be a dict which include joint
        samples and marginal samples. in this implement, the marginal is only another sample set of one distribution.
        see more details in https://arxiv.org/pdf/1801.04062.pdf
        :return: loss: the loss term
                 lb: the lower bound of mutual information (in DV mode)
        """
        joint_samples = samples['joint']
        marginal_samples = samples['marginal']

        if self.update_mode is 'DV':
            t = torch.mean(self.T(joint_samples[:, 0], joint_samples[:, 1])).to(self.device)
            et = torch.exp(self.T(joint_samples[:, 0], marginal_samples)).to(self.device)
            lb = t - torch.log(torch.mean(et))
            if self.measure is 'naive':
                loss = -lb
            elif self.measure is 'ma':
                # et use move average
                et_ma = (1 - self.ma_rate) * self.et_ma + self.ma_rate * torch.mean(et)
                self.et_ma = et_ma
                loss = -(t - (1 / et_ma.mean()).detach() * torch.mean(et))
            else:
                NotImplementedError("Check your measure! DV support naive and ma now.")
        elif self.update_mode is 'fGAN':
            Txy = self.T(joint_samples[:, 0], joint_samples[:, 1])
            Tx_y = self.T(joint_samples[:, 0], marginal_samples)
            # print(Txy)

            Ep, Eq = fGAN(Txy, Tx_y, self.measure)

            loss = - Ep.mean() + Eq.mean()
            lb = loss

        elif self.update_mode is 'NCE':
            batch_size = joint_samples.shape[0]
            joint_samples_ = joint_samples[:, 0].expand(batch_size, batch_size)
            marginal_samples_ = marginal_samples.expand(batch_size, batch_size)

            et = torch.exp(self.T(joint_samples_, marginal_samples_.T))
            Eq = torch.log(torch.sum(et, dim=0))

            loss = -(self.T(joint_samples[:, 0], joint_samples[:, 1]) - Eq).mean()
            # # print(joint_samples_)
            # # print(marginal_samples_)
            lb = loss
        else:
            NotImplementedError("Check your mutual information calculation mode! Support DV, fGAN, NCE now.")

        return loss, lb

    def update(self, samples):
        """
        :param samples: same as function "get_loss"
        :return: same as function "get_loss"
        be attention that the update function is used with only mutual information as the loss term
        you should write your own update function if you have other loss terms
        """
        # update the parameter of neural network by back propagation
        loss, lb = self.get_loss(samples)

        self.optimizer.zero_grad()
        autograd.backward(loss)
        self.optimizer.step()

        return loss, lb
