"""

This class implements the curriculum class balancing.
It implements a squared median frequency class balancing but taking both labeled and unlabeled data into account.
Unlabeled data is taken into account using pseudolabels that are updated at every iteration

"""


import numpy as np


class ClassBalancing:

    def __init__(self, labeled_iters, unlabeled_iters,  n_classes=19):
        """

        Args:
            labeled_iters: Number of iterations to fill up the memory of labeled statistics
            unlabeled_iters:  Number of iterations to fill up the memory of unlabeled statistics
            n_classes: number of classes of the dataset
        """
        self.labeled_samples = labeled_iters
        self.unlabeled_samples = unlabeled_iters
        self.n_classes = n_classes

        # build memory to store the statistcs of the labels for labeled and unlabeled data
        self.labeled_frequencies = np.zeros((labeled_iters, n_classes), dtype = np.long)
        self.unlabeled_frequencies = np.zeros((unlabeled_iters, n_classes), dtype = np.long)

        self.iter = 0 # iteration counter
        self.start_computing_iter = max(labeled_iters, unlabeled_iters) # number of iterations to take into account all statistics of the dataset


    def compute_frequencies(self, samples):
        """

        Args:
            samples: BxWxH labels or pseudolabels

        Returns: computes per-class frequencies from the input labels

        """
        freqs = np.zeros((self.n_classes))
        for c in range(self.n_classes):
            mask_freq_c = (samples == c).astype(float)
            freqs[c] = mask_freq_c.sum()
        return freqs

    def add_frequencies(self, labeled_samples, unlabeled_samples):
        """
        Given some labels and pseudolabels of an training iteration, add them to the statistics memories
        Args:
            labeled_samples: BxWxH labels
            unlabeled_samples: BxWxH pseudolabels


        """

        if self.iter < self.labeled_samples:
            labeled_freqs = self.compute_frequencies(labeled_samples)
            self.labeled_frequencies[self.iter, :] = labeled_freqs

        unl_freqs = self.compute_frequencies(unlabeled_samples)

        if self.iter < self.unlabeled_samples:
            self.unlabeled_frequencies[self.iter, :] = unl_freqs
        else: # remove first, add this one at the bottom (concat)
            # only for unlabeled because labeled doesnot change once is filled
            self.unlabeled_frequencies = self.unlabeled_frequencies[1:, :]
            self.unlabeled_frequencies = np.concatenate((self.unlabeled_frequencies, np.expand_dims(unl_freqs, 0)), axis=0)

        self.iter += 1

    def add_frequencies_labeled(self, labeled_samples):

        if self.iter < self.labeled_samples:
            labeled_freqs = self.compute_frequencies(labeled_samples)
            self.labeled_frequencies[self.iter, :] = labeled_freqs


    def add_frequencies_unlabeled(self, unlabeled_samples):

        unl_freqs = self.compute_frequencies(unlabeled_samples)

        if self.iter < self.unlabeled_samples:
            self.unlabeled_frequencies[self.iter, :] = unl_freqs
        else: # remove first, add this one at the bottom (concat)
            # only for unlabeled because labeled doesnot change
            self.unlabeled_frequencies = self.unlabeled_frequencies[1:, :]
            self.unlabeled_frequencies = np.concatenate((self.unlabeled_frequencies, np.expand_dims(unl_freqs, 0)), axis=0)

        self.iter += 1



    def get_weights(self, max_iter, only_labeled=False):
        if self.iter < self.start_computing_iter: # do not compute weights until the memories are filled up
            return np.ones((self.n_classes))
        else: # inverse median, frequency
            ratio_unlabeled = 1 # min (1., self.iter / max_iter) # weigth to give to the pseudolabels statistics
            freqs_labeled = np.sum(self.labeled_frequencies, axis = 0)
            freqs_unlabeled = np.sum(self.unlabeled_frequencies, axis = 0)
            if only_labeled:
                ratio_unlabeled = 0

            freqs = freqs_labeled + freqs_unlabeled * ratio_unlabeled

            median = np.median(freqs)
            weights = median / freqs

            # deal with classes with no samples
            mask_inf = np.isinf(weights)
            weights[mask_inf] = 1
            weights[mask_inf] = max(weights)

            return np.power(weights, 0.5)

