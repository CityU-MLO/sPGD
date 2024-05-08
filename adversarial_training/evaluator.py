from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time

import torch
from torch.nn import functional as F


class Evaluator(object):
    def __init__(self, model, attack, is_cuda=True, verbose=True, zca_trans=None):
        super(Evaluator, self).__init__()
        self.model = model
        self.attack = attack
        self.is_cuda = is_cuda
        self.verbose = verbose
        self.zca_trans = zca_trans

    def evaluate(self, data_loader, print_freq=1):
        self.model.eval()
        correct, adv_correct, total = 0, 0, 0
        start = time.time()
        for i, data in enumerate(data_loader):
            x, y = data
            if self.is_cuda:
                x = x.cuda()
                y = y.cuda()

            x_adv, _, _ = self.attack.perturb(x, y)
            # clean_acc, robust_acc, _, ind_fail = self.attack.ensemble_attack(x, y)
            with torch.no_grad():
                clean_acc = (self.model(x).max(1)[1] == y).float().sum()
                robust_acc = (self.model(x_adv).max(1)[1] == y).float().sum()
            correct += clean_acc
            adv_correct += robust_acc

            total += len(y)
            if self.verbose and (i + 1) % print_freq == 0:
                p_str = "[{:3d}|{:3d}] using {:.3f}s ...".format(
                    i + 1, len(data_loader), time.time() - start)
                print(p_str)
        return float(correct)/total, float(adv_correct)/total

