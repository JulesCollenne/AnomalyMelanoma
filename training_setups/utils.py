import datetime
import shutil

import numpy as np
import torch
from PIL import ImageFilter, Image
import random

from torch.utils.data import Dataset


def save_checkpoint(state, is_best, filename, UA):
    if UA:
        torch.save(state, filename + f"checkpoint_UA_{datetime.date.today()}.pth.tar")
    else:
        torch.save(state, filename + f"checkpoint_{datetime.date.today()}.pth.tar")
        if is_best:
            shutil.copyfile(filename, filename + f'/model_best_{datetime.date.today()}.pth.tar')


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class CustomImageDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None, dataset_path="./dataset/", feature_extraction=False,
                 get_names=False):
        self.df = df
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_path = dataset_path
        self.feature_extraction = feature_extraction
        self.get_names = get_names

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.dataset_path + self.df['image_name'].iloc[idx] + ".jpg")
        labels = self.df['target'].iloc[idx]
        names = self.df['image_name'].iloc[idx]
        label = torch.tensor(np.asarray(labels))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.feature_extraction:
            return image, names
        if self.get_names:
            return image, label, names
        return image, label


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [.1, 2.]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
