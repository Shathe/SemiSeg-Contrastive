import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
import random
from sklearn.preprocessing import  normalize as norm

def get_next_index(current_features, possible_features):
    current_features_norm = norm(current_features, axis = 1)
    possible_features_norm = norm(possible_features, axis = 1)
    similarities = np.matmul(current_features_norm, possible_features_norm.transpose())
    # N x M
    # first, per lement M, get the maximum similarity
    similarities = np.max(similarities, axis=0)

    return np.argmin(similarities)


def cosine_distance(a, b):
    return sum(norm_element(a) * norm_element(b))


def norm_element(x, axis_norm=0):
    return x / np.linalg.norm(x , axis=axis_norm)

def cosine_distance_group(a, b, axis_norm=0):

    return sum(norm(a) * norm(b))

class FeatureMemoryEma:
    def __init__(self, feature_size=256, n_classes=19):
        self.feature_size = feature_size
        self.memory = [None] * n_classes
        self.n_classes = n_classes



    def add_features_from_sample_random(self, features, class_labels, ema_update=0.998):
        features = features.detach().cpu().numpy()  # no usar gradientes
        class_labels = class_labels.detach().cpu().numpy() # no usar gradientes


        for c in range(self.n_classes):
            mask_c = class_labels == c
            features_c = features[mask_c, :]

            if features_c.shape[0] > 0:
                # TODO: new_features is the mean
                new_features = np.mean(features_c, axis=0)

                if self.memory[c] is None: # was empy, first elements
                    self.memory[c] = new_features

                else: # add elements to already existing list
                    # TODO: EMA
                    self.memory[c] = self.memory[c] * ema_update + (1 - ema_update) * new_features
