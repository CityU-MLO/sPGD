from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import kornia as K
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import datasets as D
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from imagenet100 import imagenet100
from gtsrb import load_gtsrb


def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class TensorDataset(Dataset):
    def __init__(self, images, labels, transform=None):  # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def get_data_loader(cfg, data_name='mnist', data_dir='data/mnist', batch_size=128,
                    test_batch_size=200, num_workers=4, eval_samples=-1):
    if data_name == 'mnist':
        transform_train = T.Compose([T.ToTensor()])
        transform_test = T.Compose([T.ToTensor()])
        train_set = D.MNIST(root=data_dir, train=True, download=True,
                            transform=transform_train)
        test_set = D.MNIST(root=data_dir, train=False, download=True,
                           transform=transform_test)
        img_size, num_class = 28, 10
    elif data_name == 'cifar10':
        transform_train = T.Compose([T.RandomCrop(32, padding=4),
                                     T.RandomHorizontalFlip(),
                                     T.ToTensor()
                                     ])
        transform_test = T.Compose([T.ToTensor()
                                    ])

        train_set = D.CIFAR10(root=data_dir, train=True, download=True,
                              transform=transform_train)
        test_set = D.CIFAR10(root=data_dir, train=False, download=True,
                             transform=transform_test)
        img_size, num_class = 32, 10
    elif data_name == 'cifar100':
        transform_train = T.Compose([T.RandomCrop(32, padding=4),
                                     T.RandomHorizontalFlip(),
                                     T.ToTensor()
                                     ])
        transform_test = T.Compose([T.ToTensor()
                                    ])

        train_set = D.CIFAR100(root=data_dir, train=True, download=True,
                              transform=transform_train)
        test_set = D.CIFAR100(root=data_dir, train=False, download=True,
                             transform=transform_test)
        img_size, num_class = 32, 100
    elif data_name == 'stl10':
        transform_train = T.Compose([T.RandomCrop(96, padding=4),
                                     T.RandomHorizontalFlip(),
                                     T.ToTensor(), ])
        transform_test = T.Compose([T.ToTensor()])
        train_set = D.STL10(root=data_dir, split='train', download=True,
                            transform=transform_train)
        test_set = D.STL10(root=data_dir, split='test', download=True,
                           transform=transform_test)
        img_size, num_class = 96, 10
    elif data_name == 'imagenet-sub':
        transform_train = T.Compose([
            T.RandomResizedCrop(64, scale=(0.8, 0.9), ratio=(1.0, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(), ])
        transform_test = T.Compose([T.Resize(64), T.ToTensor()])
        train_set = D.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
        test_set = D.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
        img_size, num_class = 64, 143
    elif data_name == 'imagenet100':
        train_loader, valid_loader, test_loader, num_classes = imagenet100(data_dir=data_dir, batch_size=batch_size)
        img_size, num_class = 224, 100
        return train_loader, test_loader, num_class, img_size, train_loader.dataset, test_loader.dataset
    elif data_name == 'gtsrb':
        train_loader, test_loader, num_classes, img_size, train_set, test_set = load_gtsrb(
            data_dir, batch_size, test_batch_size=cfg.eval_batch_size, eval_samples=cfg.eval_samples, num_workers=4)
        return train_loader, test_loader, num_classes, img_size, train_set, test_set
    else:
        raise ValueError('invalid dataset, current only support {}'.format(
            "mnist, cifar10, stl10, imagenet-sub"))

    if eval_samples != -1 and eval_samples < len(test_set):
        test_set = torch.utils.data.Subset(test_set, np.arange(eval_samples))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, num_class, img_size, train_set, test_set


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_model(model, path):
    if not os.path.isfile(path):
        raise IOError('model: {} is non-exists'.format(path))
    if hasattr(model, 'module'):
        module = model.module
    else:
        module = model
    checkpoint = torch.load(path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    module.load_state_dict(state_dict, strict=False)
    print('Params Loaded from: {}'.format(path))


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)