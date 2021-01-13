import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time


class CurriculumClassBalancing:

    def __init__(self, ramp_up = 10000, labeled_samples=0, unlabeled_samples=0,  n_classes=19):
        self.labeled_samples = labeled_samples
        self.unlabeled_samples = unlabeled_samples
        self.n_classes = n_classes
        self.labeled_frequencies = np.zeros((labeled_samples, n_classes), dtype = np.long)
        self.unlabeled_frequencies = np.zeros((unlabeled_samples, n_classes), dtype = np.long)
        self.iter = 0
        self.percentage_unlabeled = float(unlabeled_samples) / float(labeled_samples + unlabeled_samples)
        self.rampu_up = max(labeled_samples, unlabeled_samples)


    def compute_frequencies(self, samples, confidences=None, power = 9):
        freqs = np.zeros((self.n_classes))
        for c in range(self.n_classes):
            mask_freq_c = (samples == c).astype(float)
            if confidences is not None:
                mask_freq_c = mask_freq_c * (confidences ** power)
            freqs[c] = mask_freq_c.sum()
        return freqs

    def add_frequencies(self, labeled_samples, unlabeled_samples, unlabeled_confidences=None):

        if self.iter < self.labeled_samples:
            labeled_freqs = self.compute_frequencies(labeled_samples)
            self.labeled_frequencies[self.iter, :] = labeled_freqs

        unl_freqs = self.compute_frequencies(unlabeled_samples, unlabeled_confidences)

        if self.iter < self.unlabeled_samples:
            self.unlabeled_frequencies[self.iter, :] = unl_freqs
        else: # remove first, add this one at the bottom (concat)
            # only for unlabeled because labeled doesnot change
            self.unlabeled_frequencies = self.unlabeled_frequencies[1:, :]
            self.unlabeled_frequencies = np.concatenate((self.unlabeled_frequencies, np.expand_dims(unl_freqs, 0)), axis=0)

        self.iter += 1



    def get_weights(self, max_iter, only_labeled=False, reduction_freqs = np.sum):
        if self.iter < self.rampu_up:
            return np.ones((self.n_classes)) # in order to get all the statistics from one epoch
        else: # inverse median, frequency
            ratio_unlabeled = min (1., self.iter / max_iter)
            freqs_labeled = np.sum(self.labeled_frequencies, axis = 0)
            freqs_unlabeled = np.sum(self.unlabeled_frequencies, axis = 0)
            if only_labeled:
                ratio_unlabeled = 0

            freqs = freqs_labeled + freqs_unlabeled * ratio_unlabeled

            median = np.median(freqs)
            weights = median / freqs
            mask_inf = np.isinf(weights) # not samples on some classes

            weights[mask_inf] = 1
            weights[mask_inf] = max(weights)
            return np.power(weights, 0.5)

