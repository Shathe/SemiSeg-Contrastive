import numpy as np
import kornia
import torch
import random
import torch.nn as nn
import torchvision
from PIL import Image

def normalize(data, dataset):
    if dataset == 'pascal_voc':
        mean = (122.67891434, 116.66876762, 104.00698793)
    elif dataset == 'cityscapes':
        mean = (73.15835921, 82.90891754, 72.39239876)

    mean = torch.Tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
    data_norm = ((data-mean)/255.0)
    return data_norm


def grayscale(grayscale,  data = None, target = None, probs = None):
    # s is the strength of colorjitter
    if not (data is None):
        if grayscale and data.shape[1]==3:
            seq = nn.Sequential(kornia.augmentation.RandomGrayscale(p=1.) )
            data = seq(data)
    return data, target, probs

def colorJitter(colorJitter,  data = None, target = None, s=0.1, probs = None):
    if not (data is None):
        if colorJitter and data.shape[1]==3:
            seq = nn.Sequential(kornia.augmentation.ColorJitter(brightness=s,contrast=s,saturation=s/2.,hue=s/3.))
            data = seq(data/255.)*255. # assumes norm [0,1]
    return data, target, probs

def gaussian_blur(blur, data = None, target = None, min_sigma=0.2, max_sigma=3, probs = None):
    if not (data is None):
        if blur and data.shape[1]==3:
            seq = nn.Sequential(kornia.filters.GaussianBlur2d(kernel_size=(23, 23), sigma=(min_sigma, max_sigma)))
            data = seq(data)
    return data, target, probs

def flip(flip, data = None, target = None, probs = None):
    #Flip
    if flip:
        if not (data is None): data = torch.flip(data,(3,))#np.array([np.fliplr(data[i]).copy() for i in range(np.shape(data)[0])])
        if not (target is None):
            target = torch.flip(target,(2,))#np.array([np.fliplr(target[i]).copy() for i in range(np.shape(target)[0])])
        if not (probs is None):
            probs = torch.flip(probs,(2,))#np.array([np.fliplr(target[i]).copy() for i in range(np.shape(target)[0])])
    return data, target, probs

def solarize(solarize,  data = None, target = None, probs = None):
    # s is the strength of colorjitter
    #solarize
    if not (data is None):
        if solarize and data.shape[1]==3:
            seq = nn.Sequential(kornia.augmentation.RandomSolarize((0, 1)))
            data = seq(data.cpu()/255.).cuda()*255.
    return data, target, probs



def mix(mask, data = None, target = None, probs = None):
    #Mix
    if not (data is None):
        if mask.shape[0] == data.shape[0]:
            data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])
        elif mask.shape[0] == data.shape[0] / 2:
            data = torch.cat((torch.cat([(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))]),
                              torch.cat([((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))])))
    if not (target is None):
        target = torch.cat([(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])

    if not (probs is None):
        probs = torch.cat([(mask[i] * probs[i] + (1 - mask[i]) * probs[(i + 1) % probs.shape[0]]).unsqueeze(0) for i in range(probs.shape[0])])

    return data, target, probs


def mix2(mask, data = None, target = None, probs = None):
    #Mix
    if not (data is None):
        if mask.shape[0] == data.shape[0]:
            data = torch.cat([((1 - mask[(i + 1) % data.shape[0]]) * data[i] + mask[(i + 1) % data.shape[0]] * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])

    if not (target is None):
        target = torch.cat([((1 - mask[(i + 1) % data.shape[0]]) * target[i] + mask[(i + 1) % data.shape[0]] * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])

    if not (probs is None):
        probs = torch.cat([((1 - mask[(i + 1) % data.shape[0]]) * probs[i] + mask[(i + 1) % data.shape[0]] * probs[(i + 1) % probs.shape[0]]).unsqueeze(0) for i in range(probs.shape[0])])

    return data, target, probs


def random_scale_crop(scale, data = None, target = None, ignore_label=255, probs = None):
    if scale != 1:
        init_size_w = data.shape[2]
        init_size_h = data.shape[3]

        data = nn.functional.interpolate(data, scale_factor=scale, mode='bilinear', align_corners=True, recompute_scale_factor=True)
        if target is not None:
            target = nn.functional.interpolate(target.unsqueeze(1).float(), scale_factor=scale, mode='nearest', recompute_scale_factor=True).long().squeeze(1)
        if probs is not None:
            probs = nn.functional.interpolate(probs.unsqueeze(1), scale_factor=scale, mode='bilinear', align_corners=True, recompute_scale_factor=True).squeeze(1)

        final_size_w = data.shape[2]
        final_size_h = data.shape[3]
        diff_h = init_size_h - final_size_h
        diff_w = init_size_w - final_size_w
        if scale < 1: # pad
            if diff_h % 2 == 1:
                pad = nn.ConstantPad2d((diff_w//2, diff_w//2+1, diff_h//2+1, diff_h//2), 0)
            else:
                pad = nn.ConstantPad2d((diff_w//2, diff_w//2, diff_h//2, diff_h//2), 0)

            data = pad(data)
            if probs is not None:
                probs = pad(probs)
            if diff_h % 2 == 1:
                pad = nn.ConstantPad2d((diff_w//2, diff_w//2+1, diff_h//2+1, diff_h//2), ignore_label)
            else:
                pad = nn.ConstantPad2d((diff_w//2, diff_w//2, diff_h//2, diff_h//2), ignore_label)

            if target is not None:
                target = pad(target)
        else: # crop
            #to facilitate contrastive learning, center crop
            # w = random.randint(0, data.shape[2] - init_size_w)
            # h = random.randint(0, data.shape[3] - init_size_h)
            w = int(round((data.shape[2] - init_size_w)/2))
            h = int(round((data.shape[3] - init_size_h)/2))
            data = data [:,:,h:h+init_size_h,w:w + init_size_w]
            if probs is not None:
                probs = probs [:,h:h+init_size_h,w:w + init_size_w]
            if target is not None:
                target = target [:,h:h+init_size_h,w:w + init_size_w]

    return data, target, probs


