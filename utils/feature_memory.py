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

class FeatureMemory:
    def __init__(self, num_samples, dataset,  memory_per_class=2048, feature_size=256, n_classes=19):
        self.num_samples = num_samples
        self.memory_per_class = memory_per_class
        self.feature_size = feature_size
        self.memory = [None] * n_classes
        self.n_classes = n_classes
        if dataset == 'cityscapes': # usually all classes in one image
            self.per_class_samples_per_image = int(round(memory_per_class / num_samples))
        elif dataset == 'pascal_voc': # usually only 2/4 classes on each image, except background class
            self.per_class_samples_per_image = int(n_classes / 4 * round(memory_per_class / num_samples))



    def add_features_from_sample_learned(self, model, features, class_labels, batch_size):
        features = features.detach()  # no usar gradientes
        class_labels = class_labels.detach().cpu().numpy() # no usar gradientes

        elements_per_class = batch_size * self.per_class_samples_per_image
        # for each class, save [elements_per_class]
        for c in range(self.n_classes):
            mask_c = class_labels == c
            selector = model.__getattr__('contrastive_class_selector_' + str(c))
            features_c = features[mask_c, :]
            if features_c.shape[0] > 0: # elements of class c in batch
                if features_c.shape[0] > elements_per_class:
                    with torch.no_grad():

                        trainablility = selector(features_c)  # detach for trainability
                        trainablility = torch.sigmoid(trainablility) # torch.Size([2833, 1])
                        _, indices = torch.sort(trainablility[:, 0], dim=0)
                        indices = indices.cpu().numpy()
                        features_c = features_c.cpu().numpy()
                        features_c = features_c[indices, :]
                        new_features = features_c[:elements_per_class, :]
                else:
                    new_features = features_c.cpu().numpy()

                if self.memory[c] is None: # was empy, first elements
                    self.memory[c] = new_features

                else: # add elements to already existing list
                    # keep only most recent memory_per_class samples
                    self.memory[c] = np.concatenate((new_features, self.memory[c]), axis = 0)[:self.memory_per_class, :]



    def add_features_from_sample_learned_oneselector(self, model, features, class_labels, batch_size):
        features = features.detach()  # no usar gradientes
        class_labels = class_labels.detach().cpu().numpy() # no usar gradientes
        selector = model.selector

        elements_per_class = batch_size * self.per_class_samples_per_image
        # for each class, save [elements_per_class]
        for c in range(self.n_classes):
            mask_c = class_labels == c
            features_c = features[mask_c, :]
            if features_c.shape[0] > 0: # elements of class c in batch
                if features_c.shape[0] > elements_per_class:
                    with torch.no_grad():

                        trainablility = selector(features_c)  # detach for trainability
                        trainablility = torch.sigmoid(trainablility) # torch.Size([2833, 1])
                        _, indices = torch.sort(trainablility[:, 0], dim=0)
                        indices = indices.cpu().numpy()
                        features_c = features_c.cpu().numpy()
                        features_c = features_c[indices, :]
                        new_features = features_c[:elements_per_class, :]
                else:
                    new_features = features_c.cpu().numpy()

                if self.memory[c] is None: # was empy, first elements
                    self.memory[c] = new_features

                else: # add elements to already existing list
                    # keep only most recent memory_per_class samples
                    self.memory[c] = np.concatenate((new_features, self.memory[c]), axis = 0)[:self.memory_per_class, :]



    def add_features_from_sample_random(self, features, class_labels, batch_size):
        features = features.detach().cpu().numpy()  # no usar gradientes
        class_labels = class_labels.detach().cpu().numpy() # no usar gradientes

        elements_per_class = batch_size * self.per_class_samples_per_image
        # for each class, save [elements_per_class]
        for c in range(self.n_classes):
            mask_c = class_labels == c
            features_c = features[mask_c, :]
            if features_c.shape[0] > 0: # elements of class c in batch
                new_features = np.expand_dims(features_c[0, :], 0)

                for i in range(min(features_c.shape[0], elements_per_class - 1)):
                    new_index = random.randint(0, features_c.shape[0] - 1)

                    new_features = np.concatenate((new_features, np.expand_dims(features_c[new_index, :], 0)))
                    # remove new_index form features_c
                    features_c = np.delete(features_c, new_index, axis=0)

                # # shuffle elements
                # indexes = np.arange(features_c.shape[0])
                # np.random.shuffle(indexes)
                # features_c = features_c[indexes, :]
                # new_features = features_c[:elements_per_class, :]

                if self.memory[c] is None: # was empy, first elements
                    self.memory[c] = new_features

                else: # add elements to already existing list
                    # keep only most recent memory_per_class samples
                    self.memory[c] = np.concatenate((new_features, self.memory[c]), axis = 0)[:self.memory_per_class, :]



    def add_features_from_sample_highres(self, features, class_labels, batch_size):
        features = features.detach().cpu().numpy()  # no usar gradientes
        class_labels = class_labels.detach().cpu().numpy() # no usar gradientes

        elements_per_class = batch_size * self.per_class_samples_per_image

        # for each class, save [elements_per_class]
        for c in range(self.n_classes):
            mask_c = class_labels == c
            features_c = features[mask_c, :]
            if features_c.shape[0] > 0: # elements of class c in batch

                # shuffle elements
                indexes = np.arange(features_c.shape[0])
                np.random.shuffle(indexes)
                features_c = features_c[indexes, :]
                new_features = features_c[:elements_per_class, :]

                if self.memory[c] is None: # was empy, first elements
                    self.memory[c] = new_features

                else: # add elements to already existing list
                    # keep only most recent memory_per_class samples
                    self.memory[c] = np.concatenate((new_features, self.memory[c]), axis = 0)[:self.memory_per_class, :]

    def add_features_from_sample(self, features, class_labels, batch_size):
        features = features.detach().cpu().numpy()  # no usar gradientes
        class_labels = class_labels.detach().cpu().numpy() # no usar gradientes

        elements_per_class = batch_size * self.per_class_samples_per_image

        # for each class, save [elements_per_class]
        for c in range(self.n_classes):
            mask_c = class_labels == c
            features_c = features[mask_c, :]
            if features_c.shape[0] > 0: # elements of class c in batch
                new_features = np.expand_dims(features_c[0, :], 0)

                for i in range(min(features_c.shape[0], elements_per_class - 1)):
                    if random.random() > 0.5:
                        new_index = get_next_index(new_features, features_c)
                    else:
                        new_index = random.randint(0, features_c.shape[0] - 1)

                    new_features = np.concatenate((new_features, np.expand_dims(features_c[new_index, :], 0)))
                    # remove new_index form features_c
                    features_c = np.delete(features_c, new_index, axis=0)

                # # shuffle elements
                # indexes = np.arange(features_c.shape[0])
                # np.random.shuffle(indexes)
                # features_c = features_c[indexes, :]
                # new_features = features_c[:elements_per_class, :]

                if self.memory[c] is None: # was empy, first elements
                    self.memory[c] = new_features

                else: # add elements to already existing list
                    # keep only most recent memory_per_class samples
                    self.memory[c] = np.concatenate((new_features, self.memory[c]), axis = 0)[:self.memory_per_class, :]


        # # insert first element
        # new_f = np.expand_dims(features[0, :], axis=0)
        # new_c = np.expand_dims(class_distribution[0, :], axis=0)
        # # TOOD:e sot ya no sera * batch_size
        # maximum_elements_to_compare = min (maximum_elements_to_compare_per_image * batch_size, features.shape[0])
        # print(class_distribution[:10, :])
        # time.sleep(1000)
        #
        # self.cache_label_c = [False] * self.n_classes
        # for i in range(1, maximum_elements_to_compare):
        #     f = features[i,:]
        #     c = class_distribution[i, :]
        #     if self.check_if_insert_element(c, new_c, repeated_times = batch_size):
        #         new_f = np.concatenate((new_f, np.expand_dims(f, axis=0)), axis=0)
        #         new_c = np.concatenate((new_c, np.expand_dims(c, axis=0)), axis=0)
        #


    # def check_if_insert_element(self, class_distribution, class_distribution_list, repeated_times = 0):
    #
    #     is_similar_distribution = False
    #
    #     i = 0
    #     while i < class_distribution_list.shape[0] and not is_similar_distribution:
    #
    #         similarity = cosine_distance(class_distribution, class_distribution_list[i, :])
    #         if similarity > 0.95 and repeated_times > 0: # add only if similarity < 0.9
    #             repeated_times -= 1
    #
    #         elif similarity > 0.95: # add only if similarity < 0.9
    #             is_similar_distribution = True
    #
    #         i = i + 1
    #
    #     return not is_similar_distribution
