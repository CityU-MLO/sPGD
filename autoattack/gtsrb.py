import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, Sampler
from PIL import Image


class GTSRB(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=True, seed=0, test_size=1000, out_index=False):
        super(GTSRB, self).__init__()
        self.images = []
        self.labels = []
        self.transform = transform
        self.target_transform = target_transform
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.samples = []
        self.imgs = []
        self.targets = []
        self.root = root
        self._load_data()
        self.train = train
        self.seed = seed
        self.out_index = out_index
        # manually split training and test set
        np.random.seed(seed)
        indices = np.arange(len(self.samples))
        np.random.seed(seed)
        np.random.shuffle(indices)
        if test_size < 1:
            test_size = int(test_size * len(self.samples))
        if train:
            indices = indices[test_size:]
        else:
            indices = indices[:test_size]

        self.samples = [self.samples[i] for i in indices]
        self.imgs = [self.imgs[i] for i in indices]
        self.targets = [self.targets[i] for i in indices]

    def _load_data(self):
        self.classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}
        for target_class in self.classes:
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self.root, target_class)
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    if fname.endswith('.ppm'):
                        path = os.path.join(root, fname)
                        item = (path, class_index)
                        self.samples.append(item)
                        self.imgs.append(path)
                        self.targets.append(class_index)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.out_index:
            return img, target, index
        return img, target

    def __len__(self):
        return len(self.samples)


def load_gtsrb(root, batch_size, test_batch_size, num_workers, pin_memory=False, shuffle=True, eval_samples=-1, out_index=False):
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    dataset_train = GTSRB(root, transform=transform_train, train=True, out_index=out_index)
    dataset_test = GTSRB(root, transform=transform_test, train=False, out_index=out_index)

    if eval_samples != -1 and eval_samples < len(dataset_test):
        dataset_test = torch.utils.data.Subset(dataset_test, np.arange(eval_samples))

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers, pin_memory=pin_memory)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=pin_memory)
    return loader_train, loader_test, len(dataset_train.classes), 224, dataset_train, dataset_test