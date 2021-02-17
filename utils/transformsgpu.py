'''
Code taken from https://github.com/WilhelmT/ClassMix
Slightly modified
'''

import kornia
import torch
import random
import torch.nn as nn


def normalize_rgb(data, dataset):
    """

    Args:
        data: data to normalize BxCxWxH
        dataset: name of the dataset to normalize

    Returns:
        normalized data as  (x-mean)/255

    """
    if dataset == 'pascal_voc':
        mean = (122.6789143, 116.66876762, 104.00698793) # rgb
    elif dataset == 'cityscapes':
        mean = (73.15835921, 82.90891754, 72.39239876) # rgb
    else:
        mean = (127.5, 127.5, 127.5 )

    mean = torch.Tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
    data_norm = ((data-mean)/255.0)
    return data_norm


def normalize_bgr(data, dataset):
    """

    Args:
        data: data to normalize BxCxWxH
        dataset: name of the dataset to normalize

    Returns:
        normalized data as  (x-mean)/255

    """
    if dataset == 'pascal_voc':
        mean = (104.00698793, 116.66876762,  122.6789143) # bgr
    elif dataset == 'cityscapes':
        mean = (72.39239876, 82.90891754, 73.15835921) # bgr
    else:
        mean = (127.5, 127.5, 127.5 )

    mean = torch.Tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
    data_norm = ((data-mean)/255.0)
    return data_norm



def grayscale(grayscale,  data = None, target = None, probs = None):
    """

    Args:
        grayscale: boolean whether to apply grayscale augmentation
        data:  input data to augment BxCxWxH
        target: labels to augment BxWxH
        probs: probability masks to augment BxCxWxH

    Returns:
        data is converted from rgb to grayscale if [grayscale] is True
        target and probs are also returned with no modifications applied

    """
    if not (data is None):
        if grayscale and data.shape[1]==3:
            seq = nn.Sequential(kornia.augmentation.RandomGrayscale(p=1.) )
            data = seq(data)
    return data, target, probs

def colorJitter(colorJitter,  data = None, target = None, s=0.1, probs = None):
    """

    Args:
        colorJitter: boolean whether to apply colorJitter augmentation
        data:  input data to augment BxCxWxH
        target: labels to augment BxWxH
        probs: probability masks to augment BxCxWxH
        s: brightness and contrast strength of the color jitter

    Returns:
        colorJitter is applied to data if [colorJitter] is True
        target and probs are also returned with no modifications applied


    """
    if not (data is None):
        if colorJitter and data.shape[1]==3:
            seq = nn.Sequential(kornia.augmentation.ColorJitter(brightness=s,contrast=s,saturation=s/2.,hue=s/3.))
            data = seq(data/255.)*255. # assumes [0,1]
    return data, target, probs

def gaussian_blur(blur, data = None, target = None, min_sigma=0.2, max_sigma=3, probs = None):
    """

    Args:
        blur: boolean whether to apply blur
        data:  input data to augment BxCxWxH
        target: labels to augment BxWxH
        probs: probability masks to augment BxCxWxH
        min_sigma: minimum sigma value for the gaussian  blur
        max_sigma:  maximum sigma value for the gaussian  blur

    Returns:
        gaussian blur is applied to data if [blur] is True
        target and probs are also returned with no modifications applied

    """
    if not (data is None):
        if blur and data.shape[1]==3:
            seq = nn.Sequential(kornia.filters.GaussianBlur2d(kernel_size=(23, 23), sigma=(min_sigma, max_sigma)))
            data = seq(data)
    return data, target, probs

def flip(flip, data = None, target = None, probs = None):
    """

    Args:
        flip: boolean whether to apply flip augmentation
        data:  input data to augment BxCxWxH
        target: labels to augment BxWxH
        probs: probability masks to augment BxCxWxH

    Returns:
        data, target and probs are flipped if the boolean flip is True

    """
    if flip:
        if not (data is None): data = torch.flip(data,(3,))
        if not (target is None):
            target = torch.flip(target,(2,))
        if not (probs is None):
            probs = torch.flip(probs,(2,))
    return data, target, probs

def solarize(solarize,  data = None, target = None, probs = None):
    """

    Args:
        solarize: boolean whether to apply solarize augmentation
        data:  input data to augment BxCxWxH
        target: labels to augment BxWxH
        probs: probability masks to augment BxCxWxH

    Returns:
        data, target, probs, where
        data is solarized  if [solarize] is True

    """
    if not (data is None):
        if solarize and data.shape[1]==3:
            seq = nn.Sequential(kornia.augmentation.RandomSolarize((0, 1)))
            data = seq(data.cpu()/255.).cuda()*255.
    return data, target, probs




def mix(mask, data = None, target = None, probs = None):
    """
    Applies classMix augmentation:
    https://openaccess.thecvf.com/content/WACV2021/papers/Olsson_ClassMix_Segmentation-Based_Data_Augmentation_for_Semi-Supervised_Learning_WACV_2021_paper.pdf
    Args:
        mask: masks for applying ClassMix. A list of B elements of CxWxH tensors
        data:  input data to augment BxCxWxH
        target: labels to augment BxWxH
        probs: probability masks to augment BxCxWxH

    Returns:
         data, target and probs augmented with classMix

    """
    if not (data is None):
        if mask.shape[0] == data.shape[0]:
            data = torch.cat([((1 - mask[(i + 1) % data.shape[0]]) * data[i] + mask[(i + 1) % data.shape[0]] * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])

    if not (target is None):
        target = torch.cat([((1 - mask[(i + 1) % data.shape[0]]) * target[i] + mask[(i + 1) % data.shape[0]] * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])

    if not (probs is None):
        probs = torch.cat([((1 - mask[(i + 1) % data.shape[0]]) * probs[i] + mask[(i + 1) % data.shape[0]] * probs[(i + 1) % probs.shape[0]]).unsqueeze(0) for i in range(probs.shape[0])])

    return data, target, probs


def random_scale_crop(scale, data = None, target = None, ignore_label=255, probs = None):
    """

    Args:
        scale: scale ratio. Float
        data:  input data to augment BxCxWxH
        target: labels to augment BxWxH
        probs: probability masks to augment BxCxWxH
        ignore_label: integeer value that defines the ignore class in the datasets for the labels

    Returns:
         data, target and prob, after applied a scaling operation. output resolution is preserve as the same as the input resolution  WxH
    """
    if scale != 1:
        init_size_w = data.shape[2]
        init_size_h = data.shape[3]

        # scale data, labels and probs
        data = nn.functional.interpolate(data, scale_factor=scale, mode='bilinear', align_corners=True, recompute_scale_factor=True)
        if target is not None:
            target = nn.functional.interpolate(target.unsqueeze(1).float(), scale_factor=scale, mode='nearest', recompute_scale_factor=True).long().squeeze(1)
        if probs is not None:
            probs = nn.functional.interpolate(probs.unsqueeze(1), scale_factor=scale, mode='bilinear', align_corners=True, recompute_scale_factor=True).squeeze(1)

        final_size_w = data.shape[2]
        final_size_h = data.shape[3]
        diff_h = init_size_h - final_size_h
        diff_w = init_size_w - final_size_w
        if scale < 1: # add padding if needed
            if diff_h % 2 == 1:
                pad = nn.ConstantPad2d((diff_w//2, diff_w//2+1, diff_h//2+1, diff_h//2), 0)
            else:
                pad = nn.ConstantPad2d((diff_w//2, diff_w//2, diff_h//2, diff_h//2), 0)

            data = pad(data)
            if probs is not None:
                probs = pad(probs)

            # padding with ignore label to add to labels
            if diff_h % 2 == 1:
                pad = nn.ConstantPad2d((diff_w//2, diff_w//2+1, diff_h//2+1, diff_h//2), ignore_label)
            else:
                pad = nn.ConstantPad2d((diff_w//2, diff_w//2, diff_h//2, diff_h//2), ignore_label)

            if target is not None:
                target = pad(target)

        else: # crop if needed
            w = random.randint(0, data.shape[2] - init_size_w)
            h = random.randint(0, data.shape[3] - init_size_h)
            data = data [:,:,h:h+init_size_h,w:w + init_size_w]
            if probs is not None:
                probs = probs [:,h:h+init_size_h,w:w + init_size_w]
            if target is not None:
                target = target [:,h:h+init_size_h,w:w + init_size_w]

    return data, target, probs


