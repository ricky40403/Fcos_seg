import random
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):        
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

    def __repr__(self):
        format_str = self.__class__.__name__ + '('
        for t in self.transforms:
            format_str += '\n'
            format_str += f'    {t}'
        format_str += '\n)'

        return format_str

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.hflip(img)
            target = target.transpose(0)   

        return img, target

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        img = F.normalize(img, mean=self.mean, std=self.std)

        return img, target

class ToTensor:
    def __call__(self, img, target):
        return F.to_tensor(img), target

class ToTorch:
    def __call__(self, img, target):
        # print(target)
        img = torch.from_numpy(np.asarray(img).transpose((2, 0, 1))).float()
        # print(img.size())
        return img, target.to_torch()


class Resize:
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)

        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, img_size):
        w, h = img_size
        size = random.choice(self.min_size)
        max_size = self.max_size

        if max_size is not None:
            min_orig = float(min((w, h)))
            max_orig = float(max((w, h)))

            if max_orig / min_orig * size > max_size:
                size = int(round(max_size * min_orig / max_orig))

        if (w <= h and w == size) or (h <= w and h == size):
            return h, w

        if w < h:
            ow = size
            oh = int(size * h / w)

        else:
            oh = size
            ow = int(size * w / h)

        return oh, ow

    def __call__(self, img, target):
        
        size = self.get_size(img.size)        
        img = F.resize(img, size)        
        target = target.resize(img.size)
        
        

        return img, target
    
    
def get_transform(cfg, train = True):
    
    if train:
        if cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0] == -1:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN

        else:
            min_size = list(
                range(
                    cfg.INPUT.MIN_SIZE_TRAIN[0], cfg.INPUT.MIN_SIZE_TRAIN[1] + 1
                )
            )

        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip = 0.5

    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip = 0
    
    normalize = Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    
    transform = Compose(
        # [Resize(min_size, max_size), RandomHorizontalFlip(0.5), ToTorch(), normalize])
        # [Resize(min_size, max_size), ToTorch(), normalize])
        [Resize(min_size, max_size), RandomHorizontalFlip(flip), ToTensor(), normalize]
    )
    
    
    
    return transform