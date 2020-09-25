# -*- coding: UTF-8 -*-
"""=================================================
@IDE     :Pycharm
@Author  :Ning Wang
@Contact :@qq 574386703
@Date    :2020.9.22
@Desc    :A simple toy test for Mine
=================================================="""
import math
from Mine import MINE
import numpy as np
import matplotlib.pyplot as plt
import torch

# define 2 correlated distributions
data = np.random.multivariate_normal(mean=[0, 0],
                                     cov=[[1, 0.4], [0.4, 1]],
                                     size=10000)

plt.scatter(x=data[:, 0], y=data[:, 1])
plt.show()


# sample from dataset
def sample_batch(data, bs):
    data_batch = {}
    rn_joint = np.random.choice(range(data.shape[0]), size=bs, replace=False)
    data_batch['joint'] = _totorch(data[rn_joint, :])
    rn_marginal = np.random.choice(range(data.shape[0]), size=bs, replace=False)
    data_batch['marginal'] = _totorch(data[rn_marginal, 1])
    return data_batch


def _totorch(x):
    """
    other type to torch.float32
    """
    return torch.tensor(x, dtype=torch.float32) \
        if type(x) is not torch.float32 else x


def ma(a, window_size=100):
    """moving average of loss plot"""
    return [np.mean(a[i:i + window_size]) for i in range(0, len(a) - window_size)]


# MI估计
lbs = []
mine = MINE(num_input=2,
            mode='DV',
            measure='naive',
            lr=1e-3,
            device='cpu')
for epoch in range(5000):
    data_batch = sample_batch(data, bs=64)
    loss, lb = mine.update(data_batch)
    lbs.append(lb.detach().item())

print(lbs)
loss_ma = ma(lbs)
plt.plot(range(len(loss_ma)), loss_ma)
plt.show()

# # compare all methods
# methods = {"DV": ["naive", "ma"],
#            "fGAN": ["GAN", "JSD", "X2", "KL", "H2"],
#            "NCE": ['info']}
#
# plt.figure(1)
# legends = []
# for mode in methods:
#     measure = methods[mode]
#     for me in measure:
#         mine = MINE(num_input=2,
#                     mode=mode,
#                     measure=me,
#                     lr=1e-3,
#                     device='cpu')
#
#         lbs = []
#         for epoch in range(1000):
#             data_batch = sample_batch(data, bs=64)
#             loss, lb = mine.update(data_batch)
#             lbs.append(lb.detach().item())
#
#         loss_ma = ma(lbs)
#         plt.plot(range(len(loss_ma)), loss_ma)
#         legends.append(mode + ": " + me)
#         print(mode + ": " + me)
#
# plt.legend(legends)
#
# # validate that D_KL = 2D_JS - log4, pay attention to signs
# plt.figure(2)
# mine_GAN = MINE(num_input=2,
#                 mode='fGAN',
#                 measure='GAN',
#                 lr=1e-3,
#                 device='cpu')
#
# mine_JSD = MINE(num_input=2,
#                 mode='fGAN',
#                 measure='JSD',
#                 lr=1e-3,
#                 device='cpu')
#
# lbs_GAN = []
# lbs_JSD = []
# for epoch in range(1000):
#     data_batch = sample_batch(data, bs=64)
#
#     _, lb_GAN = mine_GAN.update(data_batch)
#     _, lb_JSD = mine_JSD.update(data_batch)
#
#     lbs_GAN.append(lb_GAN.detach().item())
#     lbs_JSD.append(lb_JSD.detach().item())
#
# loss_GAN = ma(lbs_GAN)
# loss_JSD = ma(lbs_JSD)
#
# plt.plot(range(len(loss_GAN)), loss_GAN)
# plt.plot(range(len(loss_JSD)), loss_JSD)
# plt.plot(range(len(loss_JSD)), 2 * np.array(loss_JSD) + math.log(4))
#
# plt.legend(['GAN', 'JSD', '2*JSD-log4'])
# plt.show()
