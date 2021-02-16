'''
Code taken from https://github.com/WilhelmT/ClassMix
Slightly modified
'''

import random
import numpy as np
from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):
        img, mask = Image.fromarray(img, mode="RGB"), Image.fromarray(mask, mode="L")
        assert img.size == mask.size
        for a in self.augmentations:
            img, mask = a(img, mask)
        return np.array(img), np.array(mask, dtype=np.uint8)



class RandomCrop_city(object):  # used for results in the CVPR-19 submission
    def __init__(self, size, padding=0):
        self.size = tuple(size)
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size

        # Resize to half size
        img = img.resize((int(w/2), int(h/2)), Image.BILINEAR)
        mask = mask.resize((int(w/2), int(h/2)), Image.NEAREST)

        # Random crop to input size
        x1 = random.randint(0, int(w/2) - tw)
        y1 = random.randint(0, int(h/2) - th)

        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
        )


class RandomCrop_city_highres(object):  # used for results in the CVPR-19 submission
    def __init__(self, size, padding=0):
        self.size = tuple(size)
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size

        x1 = random.randint(0, int(w) - tw)
        y1 = random.randint(0, int(h) - th)
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
        )


class Resize_city(object):  # used for results in the CVPR-19 submission
    def __init__(self, size, padding=0):
        self.size = tuple(size)
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size

        # Resize to half size
        img = img.resize((int(w/2), int(h/2)), Image.BILINEAR)

        return img, mask


class Resize_city_highres(object):
    def __init__(self, size, padding=0):
        self.size = tuple(size)
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)


        return img, mask