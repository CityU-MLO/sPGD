from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import torch
from torch.nn import functional as F
from utils import AverageMeter, mixup_data, mixup_criterion


class Trainer(object):
    """ Trainer to train adversarial attacting model """

    def __init__(self, model, attack, optimizer, summary_writer=None,
                 print_freq=1, output_freq=1, is_cuda=True, base_lr=0.1,
                 max_epoch=100, steps=[], rate=1., loss='adv', trades_beta=1., scheduler=None, mode='rand'):
        super(Trainer, self).__init__()
        self.model = model
        self.attack = attack
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.iter = 0
        self.print_freq = print_freq
        self.output_freq = output_freq
        self.is_cuda = is_cuda
        self.base_lr = base_lr
        self.max_epoch = max_epoch
        self.steps = steps
        self.rate = rate
        assert loss in ['adv', 'trades'], 'loss should be either adv or trades'
        self.loss = loss
        self.trades_beta = trades_beta
        self.get_lr_mults()
        self.scheduler = scheduler
        self.mode = mode

    def train(self, epoch, data_loader):
        self.model.train()

        batch_time = AverageMeter()
        adv_time = AverageMeter()
        loss_meter = AverageMeter()
        adv_loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        adv_acc_meter = AverageMeter()

        if self.scheduler is None:
            self.decrease_lr(epoch)
        end = time.time()

        for i, data in enumerate(data_loader):
            if self.mode == 'rand':
                if np.random.rand() < 1/2:
                    self.attack.change_masking()

            x, y = data
            if self.is_cuda:
                x = x.cuda()
                y = y.cuda()

            # Compute Adversarial Perturbations
            t0 = time.time()
            x_adv, _, _ = self.attack.perturb(x, y)
            adv_time.update(time.time() - t0)

            if self.loss == 'adv':
                adv_loss, adv_pred = self.adv_loss(x_adv, y)
            else:
                adv_loss, adv_pred = self.trades_loss(x, x_adv, y)

            self.optimizer.zero_grad()
            adv_loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            adv_loss_meter.update(adv_loss.item())
            adv_acc = self.accuracy(adv_pred, y)
            adv_acc_meter.update(adv_acc[0].item())

            self.scheduler.step()

            if self.summary_writer is not None:
                self.summary_writer.add_scalar('adv_loss_iter', adv_loss_meter.val, self.iter)
                self.summary_writer.add_scalar('adv_acc_iter', adv_acc_meter.val, self.iter)

            if (i + 1) % self.output_freq == 0:
                with torch.no_grad():
                    pred = self.model(x)
                    loss = F.cross_entropy(pred, y)
                    loss_meter.update(loss.item())
                acc = self.accuracy(pred, y)
                acc_meter.update(acc[0].item())
                if self.summary_writer is not None:
                    self.summary_writer.add_scalar('loss_iter', loss_meter.val, self.iter)
                    self.summary_writer.add_scalar('acc_iter', acc_meter.val, self.iter)

            if (i + 1) % self.print_freq == 0:
                p_str = "Epoch:[{:>3d}][{:>3d}|{:>3d}] Time:[{:.3f}/{:.3f}] " \
                        "Loss:[{:.3f}/{:.3f}] AdvLoss:[{:.3f}/{:.3f}] " \
                        "Acc:[{:.3f}/{:.3f}] AdvAcc:[{:.3f}/{:.3f}] ".format(
                    epoch, i + 1, len(data_loader), batch_time.val,
                    adv_time.val, loss_meter.val, loss_meter.avg,
                    adv_loss_meter.val, adv_loss_meter.avg, acc_meter.val,
                    acc_meter.avg, adv_acc_meter.val, adv_acc_meter.avg)
                print(p_str)

            self.iter += 1
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('loss_epoch', loss_meter.avg, epoch)
            self.summary_writer.add_scalar('adv_loss_epoch', adv_loss_meter.avg, epoch)
            self.summary_writer.add_scalar('acc_epoch', acc_meter.avg, epoch)
            self.summary_writer.add_scalar('adv_acc_epoch', adv_acc_meter.avg, epoch)

    def adv_loss(self, x, y):
        adv_pred = self.model(x)
        adv_loss = F.cross_entropy(adv_pred, y)
        return adv_loss, adv_pred

    def trades_loss(self, x, x_adv, y):
        clean_pred = self.model(x)
        clean_loss = F.cross_entropy(clean_pred, y)
        adv_pred = self.model(x_adv)
        robust_loss = F.kl_div(F.log_softmax(adv_pred, dim=1), F.softmax(clean_pred, dim=1), reduction='batchmean')
        loss = clean_loss + self.trades_beta*robust_loss
        return loss, adv_pred

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        if target.dim() > 1:
            target = torch.argmax(target, dim=-1)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret

    def reset(self):
        self.iter = 0

    def close(self):
        self.iter = 0
        if self.summary_writer is not None:
            self.summary_writer.close()

    def decrease_lr(self, epoch):
        lr_mult = self.lr_mults[epoch]
        for g in self.optimizer.param_groups:
            g['lr'] = lr_mult * self.base_lr * g.get('lr_mult', 1.0)

    def get_lr_mults(self):
        self.lr_mults = np.ones(self.max_epoch)
        self.steps = sorted(filter(lambda x: 0 < x < self.max_epoch, self.steps))
        if len(self.steps) > 0 and 0 < self.rate < 1.:
            for step in self.steps:
                self.lr_mults[step:] *= self.rate
