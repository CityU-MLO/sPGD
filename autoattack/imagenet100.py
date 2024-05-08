import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, Sampler


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class HybridBatchSampler(Sampler):
    def __init__(self, idx4ori, idx4plus, plus_prop, batch_size, permutation):
        self.idx4ori = idx4ori
        self.idx4plus = idx4plus
        self.plus_prop = plus_prop
        self.batch_size = batch_size
        self.permutation = permutation
        self.use_plus = plus_prop > 0

    def __iter__(self, ):

        batch = []
        # No additional data
        if self.use_plus is False:
            if self.permutation is True:
                self.idx4ori = np.random.permutation(self.idx4ori)
            for idx in self.idx4ori:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0:
                yield batch
                batch = []
        else:
            max_num_plus = int(self.batch_size * self.plus_prop)
            max_num_ori = int(self.batch_size - max_num_plus)
            idx_in_ori = 0
            if self.permutation is True:
                self.idx4plus = np.random.permutation(self.idx4plus)
            for idx in self.idx4plus:
                batch.append(idx)
                if len(batch) == max_num_plus:
                    if len(self.idx4ori[idx_in_ori: idx_in_ori + max_num_ori]) != max_num_ori:
                        if self.permutation is True:
                            self.idx4ori = np.random.permutation(self.idx4ori)
                        idx_in_ori = 0
                    batch = batch + [v for v in self.idx4ori[idx_in_ori: idx_in_ori + max_num_ori]]
                    idx_in_ori += max_num_ori
                    yield batch
                    batch = []
            if len(batch) > 0:
                num_ori = int(len(batch) / self.plus_prop * (1. - self.plus_prop))
                if len(self.idx4ori[idx_in_ori: idx_in_ori + num_ori]) != num_ori:
                    if self.permutation is True:
                        self.idx4ori = np.random.permutation(self.idx4ori)
                    idx_in_ori = 0
                batch = batch + [v for v in self.idx4ori[idx_in_ori: idx_in_ori + num_ori]]
                idx_in_ori += num_ori
                yield batch
                batch = []

    def __len__(self, ):

        if self.use_plus is False:
            return len(self.idx4ori - 1) // self.batch_size + 1
        else:
            max_num_plus = int(self.batch_size * self.plus_prop)
            return len(self.idx4plus - 1) // max_num_plus + 1


class IndexedImageNet100(datasets.ImageFolder):
    def __init__(self, root, split, transform, out_index=False):
        super(IndexedImageNet100, self).__init__(root=os.path.join(root, split), transform=transform)
        self.out_index = out_index

    def __getitem__(self, index):
        image, target = super(IndexedImageNet100, self).__getitem__(index)
        if self.out_index:
            return image, target, index
        return image, target


def imagenet100(data_dir, batch_size, valid_ratio=None, shuffle=True, augmentation=True, train_subset=None, out_index=False):
    root = ''
    for data_path in [data_dir, '#SPECIFY YOUR OWN PATH HERE#']:
        if os.path.exists(data_path):
            root = data_path
            print('Data root in %s' % root)
            break

    if root == '':
        raise "Download ImageNet dataset and run ./dataset/format_imagent.py !"

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]) if augmentation else transforms.Compose([
        transforms.ToTensor()
    ])
    transform_valid = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    trainset = IndexedImageNet100(root=data_path, split='train', transform=transform_train, out_index=out_index)
    validset = IndexedImageNet100(root=data_path, split='train', transform=transform_valid, out_index=out_index)
    testset = IndexedImageNet100(root=data_path, split='val_sub', transform=transform_test, out_index=out_index)

    classes = list(range(100))

    if train_subset is None:
        instance_num = len(trainset)
        indices = np.random.permutation(list(range(instance_num)))
    else:
        indices = np.random.permutation(train_subset)
        instance_num = len(indices)
    print('%d instances are picked from the training set' % instance_num)

    if valid_ratio is not None and valid_ratio > 0.:
        split_pt = int(instance_num * valid_ratio)
        train_idx, valid_idx = indices[split_pt:], indices[:split_pt]
        if shuffle:
            train_sampler, valid_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)
        else:
            train_sampler, valid_sampler = SubsetSampler(train_idx), SubsetSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                                   num_workers=4, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, sampler=valid_sampler,
                                                   num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4,
                                                  pin_memory=True)

    else:
        if shuffle:
            train_sampler = SubsetRandomSampler(indices)
        else:
            train_sampler = SubsetSampler(indices)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                                   num_workers=4, pin_memory=True)
        valid_loader = None
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4,
                                                  pin_memory=True)

    return train_loader, valid_loader, test_loader, classes