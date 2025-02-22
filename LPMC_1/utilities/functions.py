# -*- coding: UTF-8 -*-
"""
@filename:data_manager.py
@author:Chen Kunxu
@Time:2023/8/5
"""
import numpy as np
import numpy.random as random
import torch


def normalize(data):
    a = (data - data.mean(axis=0)) / (data.std(axis=0)+1e-5)  # +0.0001 防止std为0
    x = np.isnan(a)
    a[x] = 0
    return a


def fixed_initial_net(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
