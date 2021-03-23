"""
Python implementation of the color map function for the PASCAL VOC data set.
Official Matlab version can be found in the PASCAL VOC devkit
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
"""
from __future__ import print_function
import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import cv2
import random
import scipy
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="Dataset to train", default='./out_dir/Datasets/cityscapes')
parser.add_argument("--output_dir", help="Dataset to train", default='./out_dir/Datasets/cityscapes_colored')
args = parser.parse_args()
from collections import namedtuple
input_dir = args.input_dir
output_dir = args.output_dir



def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
          'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
          'void']
nclasses = 21
row_size = 50
col_size = 500
cmap = color_map()

outputs = glob.glob(input_dir + '/*')
for output in outputs:
    name = output.split('/')[-1]
    output_name = output_dir   + name
    print(output_name)
    target = np.array(Image.open(output))[:, :, np.newaxis]
    cmap = color_map()[:, np.newaxis, :]
    new_im = np.dot(target == 0, cmap[0])
    for i in range(1, cmap.shape[0]):
        new_im += np.dot(target == i, cmap[i])
    new_im = Image.fromarray(new_im.astype(np.uint8))
    new_im.save(output_name)

