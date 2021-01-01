import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np



def pix_to_pix_prediction(prediction, projection):
    projection = projection.detach() # stop grads
    projection = F.normalize(projection, dim=1)
    prediction = F.normalize(prediction, dim=1)

    distance = 1 - (projection * prediction).sum(axis=1).mean()
    return distance


def pix_to_pix_projection(projection1, projection2):

    projection1 = F.normalize(projection1, dim=1)
    projection2 = F.normalize(projection2, dim=1)

    distance = 1 - (projection1 * projection2).sum(axis=1).mean()
    return distance



def contrastive_class_to_class_learned_oneselector(model, features, class_labels, prediction_probs, batch_size, num_classes,
                            memory, label_probs, per_class_samples_per_image=256, minimize_top_k_percent=1., detach=True):

    loss = 0
    selector = model.selector

    for c in range(num_classes):
        mask_c = class_labels == c
        features_c = features[mask_c,:]
        prediction_probs_c = prediction_probs[mask_c]


        # TODO: sort by lowest prediction probs i.e, larger error

        memory_c = memory[c] # N, 256
        if memory_c is not None and features_c.shape[0] > 1:

            memory_c = torch.from_numpy(memory_c).cuda()
            memory_c = F.normalize(memory_c, dim=1)
            features_c_norm = F.normalize(features_c, dim=1)

            # TODO: cuanto feature size se puede hcaer yendo rapido?
            similarities = torch.mm(features_c_norm, memory_c.transpose(1, 0)) # (-1, 1) 1 is equal vectors
            distances = 1 - similarities # (0, 2) 0 is equal vectors
            # M (elements), N (memory)

            # # TODO: concat  prob distr y labels probs.
            # probs = label_probs.detach()
            # probs_c = probs[mask_c]
            # features_c = torch.cat((features_c, prediction_probs_c.unsqueeze(1), probs_c.unsqueeze(1)), dim = 1)
            if detach:
                trainablility = selector(features_c.detach())  # detach for trainability
            else:
                trainablility = selector(features_c) # detach for trainability

            trainablility = torch.sigmoid(trainablility)
            rescaled_trainability = (trainablility.shape[0] / trainablility.sum(dim=0)) * trainablility
            rescaled_trainability = rescaled_trainability.repeat(1, distances.shape[1])
            distances = distances * rescaled_trainability


            # TODO: optiomz, minimiza all. Minimize top-K
            if minimize_top_k_percent < 1:
                distances, indices = torch.sort(distances, dim=1)
                top_k = int(distances.shape[1] * minimize_top_k_percent)
                distances = distances[:, :top_k]


            #TODO  : ESTO QUE LO APRENDA SOLO
            # if label_probs is not None:
            #     probs = label_probs.detach()
            #     probs_c = probs[mask_c]
            #
            #     distances = distances.mean(dim=1)
            #     distances = distances * torch.pow(probs_c, 6)


            loss = loss + distances.mean()

    return loss / num_classes


def contrastive_class_to_class_with_negatives(features, class_labels, prediction_probs, batch_size, num_classes, memory,
                                              label_probs, per_class_samples_per_image=256, minimize_top_k_percent=1.):
    elements_per_class = batch_size * per_class_samples_per_image
    '''
    La diferencai seria que tendrias qeu multiplicar con toda la memoria y luego sacar los positivos y engativos de la calase y generar la loss

    A parte de que aqui, no hay prediction head

    '''
    temp = 0.07
    positives = 0
    negatives = 0
    for c in range(num_classes):
        mask_c = class_labels == c
        features_c = features[mask_c, :]

        prediction_probs_c = prediction_probs[mask_c]

        # TODO: sort by lowest prediction probs i.e, larger error
        _, indices = torch.sort(prediction_probs_c)
        features_c = features_c[indices, :]
        random_features_c = features_c[:elements_per_class, :]  # M, 256

        # TODO: pass confidece per sample to weight loss
        if label_probs is not None:
            probs = label_probs.detach()
            probs_c = probs[mask_c]
            probs_c = probs_c[indices]
            probs_c = probs_c[:elements_per_class]

        for compare_c in range(num_classes):
            is_positive = c == compare_c
            memory_c = memory[compare_c]  # N, 256
            if memory_c is not None and random_features_c.shape[0] > 0:
                memory_c = torch.from_numpy(memory_c).cuda()
                memory_c = F.normalize(memory_c, dim=1)
                random_features_c = F.normalize(random_features_c, dim=1)

                # TODO: cuanto feature size se puede hcaer yendo rapido?
                similarities = torch.mm(random_features_c, memory_c.transpose(1, 0))  # (-1, 1) 1 is equal vectors
                # M (elements), N (memory)

                # TODO: optiomz, minimiza all. Minimize top-K
                if minimize_top_k_percent < 1:
                    # TODO: demomento esta en modo facil
                    if is_positive:  # aqui quiero quedarme con la similareidades mascercas de -1, es decir als que menos e aprezcan
                        similarities, indices = torch.sort(similarities, dim=1, descending=True)
                    else:  # aqui quedarme con los que se parezcan mucho apra seaprarlas, cuatno mas altas mejor
                        similarities, indices = torch.sort(similarities, dim=1, descending=False)
                    top_k = int(similarities.shape[1] * minimize_top_k_percent)
                    similarities = similarities[:, :top_k]

                similarities_exp = torch.exp(similarities / temp)

                # weight with confidences
                if label_probs is not None:
                    similarities_exp = similarities_exp.sum(dim=1)
                    similarities_exp = similarities_exp * torch.pow(probs_c, 9)

                similarities_exp = similarities_exp.sum()

                if is_positive:
                    positives = positives + similarities_exp
                else:
                    negatives = negatives + similarities_exp

    loss = - torch.log(positives / (negatives + positives))
    return loss

def contrastive_class_to_class(features, class_labels, prediction_probs, batch_size, num_classes,
                            memory, label_probs, per_class_samples_per_image=256, minimize_top_k_percent=1.):
    elements_per_class = batch_size * per_class_samples_per_image
    loss = 0

    for c in range(num_classes):
        mask_c = class_labels == c
        features_c = features[mask_c,:]
        prediction_probs_c = prediction_probs[mask_c]

        # TODO: sort by lowest prediction probs i.e, larger error
        _, indices = torch.sort(prediction_probs_c)
        features_c = features_c[indices, :]
        random_features_c = features_c[:elements_per_class, :] # M, 256


        memory_c = memory[c] # N, 256
        if memory_c is not None and random_features_c.shape[0] > 0:

            memory_c = torch.from_numpy(memory_c).cuda()
            memory_c = F.normalize(memory_c, dim=1)
            random_features_c = F.normalize(random_features_c, dim=1)

            # TODO: cuanto feature size se puede hcaer yendo rapido?
            similarities = torch.mm(random_features_c, memory_c.transpose(1, 0)) # (-1, 1) 1 is equal vectors
            distances = 1 - similarities # (0, 2) 0 is equal vectors
            # M (elements), N (memory)

            # TODO: optiomz, minimiza all. Minimize top-K
            if minimize_top_k_percent < 1:
                # TODO: coger  entre 0.2 y 0.8?
                distances, indices = torch.sort(distances, dim=1)
                top_k = int(distances.shape[1] * minimize_top_k_percent)
                distances = distances[:, :top_k]

                # TODO: OTHER OPTION. Hacer pruebas con los peso sentrenados con contrsative
                # near = distances[:, 0]
                # threshold = near * 2
                # threshold = threshold.unsqueeze(1).repeat(1, distances.shape[1]).detach().bool()
                # mask_threshold = (distances < threshold).float()
                # distances = distances * mask_threshold


            # TODO: pass confidece per sample to weight loss
            if label_probs is not None:
                probs = label_probs.detach()
                probs_c = probs[mask_c]
                probs_c = probs_c[indices]
                probs_c = probs_c[:elements_per_class]  # M, 256

                distances = distances.mean(dim=1)
                distances = distances * torch.pow(probs_c, 9)


            loss = loss + distances.mean()

    return loss / num_classes


def contrastive_class_to_class_basic_random(features, class_labels, prediction_probs, batch_size, num_classes,
                            memory, label_probs, per_class_samples_per_image=256, minimize_top_k_percent=1.):
    elements_per_class = batch_size * per_class_samples_per_image
    loss = 0

    for c in range(num_classes):
        mask_c = class_labels == c
        features_c = features[mask_c,:]

        indices = np.arange(features_c.shape[0])
        np.random.shuffle(indices)
        features_c = features_c[indices, :]
        random_features_c = features_c[:elements_per_class, :] # M, 256

        memory_c = memory[c] # N, 256
        if memory_c is not None and random_features_c.shape[0] > 0:

            memory_c = torch.from_numpy(memory_c).cuda()
            memory_c = F.normalize(memory_c, dim=1)
            random_features_c = F.normalize(random_features_c, dim=1)

            # TODO: cuanto feature size se puede hcaer yendo rapido?
            similarities = torch.mm(random_features_c, memory_c.transpose(1, 0)) # (-1, 1) 1 is equal vectors
            distances = 1 - similarities # (0, 2) 0 is equal vectors
            # M (elements), N (memory)

            # TODO: optiomz, minimiza all. Minimize top-K
            if minimize_top_k_percent < 1:
                # TODO: coger  entre 0.2 y 0.8?
                distances, indices = torch.sort(distances, dim=1)
                top_k = int(distances.shape[1] * minimize_top_k_percent)
                distances = distances[:, :top_k]

                # TODO: OTHER OPTION. Hacer pruebas con los peso sentrenados con contrsative
                # near = distances[:, 0]
                # threshold = near * 2
                # threshold = threshold.unsqueeze(1).repeat(1, distances.shape[1]).detach().bool()
                # mask_threshold = (distances < threshold).float()
                # distances = distances * mask_threshold


            loss = loss + distances.mean()

    return loss / num_classes


def contrastive_class_to_class_learned(model, features, class_labels, prediction_probs, batch_size, num_classes,
                            memory, label_probs, per_class_samples_per_image=256, minimize_top_k_percent=1., detach=True):

    loss = 0

    for c in range(num_classes):
        mask_c = class_labels == c
        features_c = features[mask_c,:]
        prediction_probs_c = prediction_probs[mask_c]

        selector = model.__getattr__('contrastive_class_selector_' + str(c))

        # TODO: sort by lowest prediction probs i.e, larger error

        memory_c = memory[c] # N, 256
        if memory_c is not None and features_c.shape[0] > 1:

            memory_c = torch.from_numpy(memory_c).cuda()
            memory_c = F.normalize(memory_c, dim=1)
            features_c_norm = F.normalize(features_c, dim=1)

            # TODO: cuanto feature size se puede hcaer yendo rapido?
            similarities = torch.mm(features_c_norm, memory_c.transpose(1, 0)) # (-1, 1) 1 is equal vectors
            distances = 1 - similarities # (0, 2) 0 is equal vectors
            # M (elements), N (memory)

            # # TODO: concat  prob distr y labels probs.
            # probs = label_probs.detach()
            # probs_c = probs[mask_c]
            # features_c = torch.cat((features_c, prediction_probs_c.unsqueeze(1), probs_c.unsqueeze(1)), dim = 1)
            if detach:
                trainablility = selector(features_c.detach()) # detach for trainability
            else:
                trainablility = selector(features_c) # detach for trainability

            trainablility = torch.sigmoid(trainablility)
            rescaled_trainability = (trainablility.shape[0] / trainablility.sum(dim=0)) * trainablility
            rescaled_trainability = rescaled_trainability.repeat(1, distances.shape[1])
            distances = distances * rescaled_trainability


            # TODO: optiomz, minimiza all. Minimize top-K
            if minimize_top_k_percent < 1:
                distances, indices = torch.sort(distances, dim=1)
                top_k = int(distances.shape[1] * minimize_top_k_percent)
                distances = distances[:, :top_k]


            #TODO  : ESTO QUE LO APRENDA SOLO
            # if label_probs is not None:
            #     probs = label_probs.detach()
            #     probs_c = probs[mask_c]
            #
            #     distances = distances.mean(dim=1)
            #     distances = distances * torch.pow(probs_c, 6)


            loss = loss + distances.mean()

    return loss / num_classes



def contrastive_class_to_class_learned_memory(model, features, class_labels, prediction_probs, batch_size, num_classes,
                            memory, label_probs, per_class_samples_per_image=256, minimize_top_k_percent=1., detach=True):

    loss = 0

    for c in range(num_classes):
        mask_c = class_labels == c
        features_c = features[mask_c,:]
        prediction_probs_c = prediction_probs[mask_c]

        selector = model.__getattr__('contrastive_class_selector_' + str(c))
        selector_memory = model.__getattr__('contrastive_class_selector_memory' + str(c))

        # TODO: sort by lowest prediction probs i.e, larger error

        memory_c = memory[c] # N, 256
        if memory_c is not None and features_c.shape[0] > 1:

            memory_c = torch.from_numpy(memory_c).cuda()
            trainablility_memory = selector_memory(memory_c)  # detach for trainability

            memory_c = F.normalize(memory_c, dim=1)
            features_c_norm = F.normalize(features_c, dim=1)

            # TODO: cuanto feature size se puede hcaer yendo rapido?
            similarities = torch.mm(features_c_norm, memory_c.transpose(1, 0)) # (-1, 1) 1 is equal vectors
            distances = 1 - similarities # (0, 2) 0 is equal vectors
            # M (elements), N (memory)

            # # TODO: concat  prob distr y labels probs.
            # probs = label_probs.detach()
            # probs_c = probs[mask_c]
            # features_c = torch.cat((features_c, prediction_probs_c.unsqueeze(1), probs_c.unsqueeze(1)), dim = 1)
            if detach:
                trainablility = selector(features_c.detach()) # detach for trainability
            else:
                trainablility = selector(features_c) # detach for trainability

            trainablility = torch.sigmoid(trainablility)
            rescaled_trainability = (trainablility.shape[0] / trainablility.sum(dim=0)) * trainablility
            rescaled_trainability = rescaled_trainability.repeat(1, distances.shape[1])
            distances = distances * rescaled_trainability



            trainablility_memory = torch.sigmoid(trainablility_memory)
            trainablility_memory = trainablility_memory.permute(1, 0)
            rescaled_trainability_memory = (trainablility_memory.shape[0] / trainablility_memory.sum(dim=0)) * trainablility_memory
            rescaled_trainability_memory = rescaled_trainability_memory.repeat(distances.shape[0], 1)
            distances = distances * rescaled_trainability_memory



            #TODO  : ESTO QUE LO APRENDA SOLO
            # if label_probs is not None:
            #     probs = label_probs.detach()
            #     probs_c = probs[mask_c]
            #
            #     distances = distances.mean(dim=1)
            #     distances = distances * torch.pow(probs_c, 6)


            loss = loss + distances.mean()

    return loss / num_classes


def contrastive_class_to_class_basic_random_selectmemory(model, features, class_labels, prediction_probs, batch_size, num_classes,
                            memory, label_probs, per_class_samples_per_image=256, minimize_top_k_percent=1.):
    elements_per_class = batch_size * per_class_samples_per_image
    loss = 0

    for c in range(num_classes):
        mask_c = class_labels == c
        features_c = features[mask_c,:]

        selector = model.__getattr__('contrastive_class_selector_memory' + str(c))


        indices = np.arange(features_c.shape[0])
        np.random.shuffle(indices)
        features_c = features_c[indices, :]
        random_features_c = features_c[:elements_per_class, :] # M, 256

        memory_c = memory[c] # N, 256
        if memory_c is not None and random_features_c.shape[0] > 0:

            memory_c = torch.from_numpy(memory_c).cuda()
            trainablility = selector(memory_c)  # detach for trainability

            memory_c = F.normalize(memory_c, dim=1)
            random_features_c = F.normalize(random_features_c, dim=1)

            # TODO: cuanto feature size se puede hcaer yendo rapido?
            similarities = torch.mm(random_features_c, memory_c.transpose(1, 0)) # (-1, 1) 1 is equal vectors
            distances = 1 - similarities # (0, 2) 0 is equal vectors
            # M (elements), N (memory)


            trainablility = torch.sigmoid(trainablility)
            trainablility = trainablility.permute(1, 0)
            rescaled_trainability = (trainablility.shape[0] / trainablility.sum(dim=0)) * trainablility
            rescaled_trainability = rescaled_trainability.repeat(distances.shape[0], 1)
            distances = distances * rescaled_trainability




            loss = loss + distances.mean()

    return loss / num_classes

def contrastive_class_to_class_basic_all(features, class_labels, prediction_probs, batch_size, num_classes,
                            memory, label_probs, per_class_samples_per_image=256, minimize_top_k_percent=1.):
    elements_per_class = batch_size * per_class_samples_per_image
    loss = 0

    for c in range(num_classes):
        mask_c = class_labels == c
        features_c = features[mask_c,:]

        random_features_c = features_c

        memory_c = memory[c] # N, 256
        if memory_c is not None and random_features_c.shape[0] > 0:

            memory_c = torch.from_numpy(memory_c).cuda()
            memory_c = F.normalize(memory_c, dim=1)
            random_features_c = F.normalize(random_features_c, dim=1)

            # TODO: cuanto feature size se puede hcaer yendo rapido?
            similarities = torch.mm(random_features_c, memory_c.transpose(1, 0)) # (-1, 1) 1 is equal vectors
            distances = 1 - similarities # (0, 2) 0 is equal vectors
            # M (elements), N (memory)

            # TODO: optiomz, minimiza all. Minimize top-K
            if minimize_top_k_percent < 1:
                # TODO: coger  entre 0.2 y 0.8?
                distances, indices = torch.sort(distances, dim=1)
                top_k = int(distances.shape[1] * minimize_top_k_percent)
                distances = distances[:, :top_k]

                # TODO: OTHER OPTION. Hacer pruebas con los peso sentrenados con contrsative
                # near = distances[:, 0]
                # threshold = near * 2
                # threshold = threshold.unsqueeze(1).repeat(1, distances.shape[1]).detach().bool()
                # mask_threshold = (distances < threshold).float()
                # distances = distances * mask_threshold


            loss = loss + distances.mean()

    return loss / num_classes







