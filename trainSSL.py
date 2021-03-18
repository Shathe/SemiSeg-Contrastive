import argparse
import os
import timeit
import datetime
import pickle
from contrastive_losses import *
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
import torch.nn as nn
from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted

from utils import transformmasks
from utils import transformsgpu

from data import get_loader, get_data_path
from data.augmentations import *

import json
from evaluateSSL import evaluate
from utils.class_balancing import ClassBalancing
from utils.feature_memory import *

start = timeit.default_timer()
start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')

class Learning_Rate_Object(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

def entropy_loss(v, mask):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()

    loss_image = torch.mul(v, torch.log2(v + 1e-30))
    loss_image = torch.sum(loss_image, dim=1)
    loss_image = mask.float() * loss_image


    percentage_valid_points = torch.mean(mask.float())

    return -torch.sum(loss_image) / (n * h * w * np.log2(c) * percentage_valid_points)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("-c", "--config", type=str, default='config.json',
                        help='Path to the config file (default: config.json)')
    parser.add_argument("-r", "--resume", type=str, default=None,
                        help='Path to the .pth file to resume from (default: None)')
    return parser.parse_args()


def lr_poly(base_lr, iter, max_iter, power):
    """

    Args:
        base_lr: initial learning rate
        iter: current iteration
        max_iter: maximum number of iterations
        power: power value for polynomial decay

    Returns: the updated learning rate with polynomial decay

    """

    return base_lr * ((1 - float(iter) / float(max_iter)) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    """

    Args:
        optimizer: pytorch optimizer
        i_iter: current iteration

    Returns: sets learning rate with poliynomial decay

    """
    lr = lr_poly(learning_rate, i_iter, num_iterations, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr


def sigmoid_ramp_up(iter, max_iter):
    """

    Args:
        iter: current iteration
        max_iter: maximum number of iterations to perform the rampup

    Returns:
        returns 1 if iter >= max_iter
        returns [0,1] incrementally from 0 to max_iters if iter < max_iter

    """
    if iter >= max_iter:
        return 1
    else:
        return np.exp(- 5 * (1 - float(iter) / float(max_iter)) ** 2)


def augmentationTransform(parameters, data=None, target=None, probs=None, jitter_vale=0.4, min_sigma=0.2, max_sigma=2., ignore_label=255):
    """

    Args:
        parameters: dictionary with the augmentation configuration
        data: BxCxWxH input data to augment
        target: BxWxH labels to augment
        probs: BxWxH probability map to augment
        jitter_vale:  jitter augmentation value
        min_sigma: min sigma value for blur
        max_sigma: max sigma value for blur
        ignore_label: value for ignore class

    Returns:
            augmented data, target, probs
    """
    assert ((data is not None) or (target is not None))
    if "Mix" in parameters:
        data, target, probs = transformsgpu.mix(mask=parameters["Mix"], data=data, target=target, probs=probs)

    if "RandomScaleCrop" in parameters:
        data, target, probs = transformsgpu.random_scale_crop(scale=parameters["RandomScaleCrop"], data=data,
                                                              target=target, probs=probs, ignore_label=ignore_label)
    if "flip" in parameters:
        data, target, probs = transformsgpu.flip(flip=parameters["flip"], data=data, target=target, probs=probs)

    if "ColorJitter" in parameters:
        data, target, probs = transformsgpu.colorJitter(colorJitter=parameters["ColorJitter"], data=data, target=target,
                                                        probs=probs, s=jitter_vale)
    if "GaussianBlur" in parameters:
        data, target, probs = transformsgpu.gaussian_blur(blur=parameters["GaussianBlur"], data=data, target=target,
                                                          probs=probs, min_sigma=min_sigma, max_sigma=max_sigma)

    if "Grayscale" in parameters:
        data, target, probs = transformsgpu.grayscale(grayscale=parameters["Grayscale"], data=data, target=target,
                                                      probs=probs)
    if "Solarize" in parameters:
        data, target, probs = transformsgpu.solarize(solarize=parameters["Solarize"], data=data, target=target,
                                                     probs=probs)

    return data, target, probs


def _save_checkpoint(iteration, model, optimizer, config, save_best=False, overwrite=True):
    """
    Saves the current checkpoint

    Args:
        iteration: current iteration [int]
        model: segmentation model
        optimizer: pytorch optimizer
        config: configuration
        save_best: Boolean: whether to sae only if best metric
        overwrite: whether to overwrite if ther is an existing checkpoint

    Returns:

    """
    checkpoint = {
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    checkpoint['model'] = model.state_dict()

    if save_best:
        filename = os.path.join(checkpoint_dir, f'best_model.pth')
        torch.save(checkpoint, filename)
        print(f'\nSaving a checkpoint: {filename} ...')
        print("Saving current best model: best_model.pth")
    else:
        filename = os.path.join(checkpoint_dir, f'checkpoint-iter{iteration}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(checkpoint, filename)
        if overwrite:
            try:
                os.remove(os.path.join(checkpoint_dir, f'checkpoint-iter{iteration - save_checkpoint_every}.pth'))
            except:
                pass


def create_ema_model(model, net_class):
    """

    Args:
        model: segmentation model to copy parameters from
        net_class: segmentation model class

    Returns: Segmentation model from [net_class] with same parameters than [model]

    """
    ema_model = net_class(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()

    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    """

    Args:
        ema_model: model to update
        model: model from which to update parameters
        alpha_teacher: value for weighting the ema_model
        iteration: current iteration

    Returns: ema_model, with parameters updated follwoing the exponential moving average of [model]

    """
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration*10 + 1), alpha_teacher)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]

    return ema_model

def augment_samples(images, labels, probs, do_classmix, batch_size, ignore_label, weak = False):
    """
    Perform data augmentation

    Args:
        images: BxCxWxH images to augment
        labels:  BxWxH labels to augment
        probs:  BxWxH probability maps to augment
        do_classmix: whether to apply classmix augmentation
        batch_size: batch size
        ignore_label: ignore class value
        weak: whether to perform weak or strong augmentation

    Returns:
        augmented data, augmented labels, augmented probs

    """

    if do_classmix:
        # ClassMix: Get mask for image A
        for image_i in range(batch_size):  # for each image
            classes = torch.unique(labels[image_i])  # get unique classes in pseudolabel A
            nclasses = classes.shape[0]

            # remove ignore class
            if ignore_label in classes and len(classes) > 1 and nclasses > 1:
                classes = classes[classes != ignore_label]
                nclasses = nclasses - 1

            if dataset == 'pascal_voc':  # if voc dataaset, remove class 0, background
                if 0 in classes and len(classes) > 1 and nclasses > 1:
                    classes = classes[classes != 0]
                    nclasses = nclasses - 1

            # pick half of the classes randomly
            classes = (classes[torch.Tensor(
                np.random.choice(nclasses, int(((nclasses - nclasses % 2) / 2) + 1), replace=False)).long()]).cuda()

            # acumulate masks
            if image_i == 0:
                MixMask = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()
            else:
                MixMask = torch.cat(
                    (MixMask, transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()))


        params = {"Mix": MixMask}
    else:
        params = {}

    if weak:
        params["flip"] = random.random() < 0.5
        params["ColorJitter"] = random.random() < 0.2
        params["GaussianBlur"] = random.random() < 0.
        params["Grayscale"] = random.random() < 0.0
        params["Solarize"] = random.random() < 0.0
        if random.random() < 0.5:
            scale = random.uniform(0.75, 1.75)
        else:
            scale = 1
        params["RandomScaleCrop"] = scale

        # Apply strong augmentations to unlabeled images
        image_aug, labels_aug, probs_aug = augmentationTransform(params,
                                                                 data=images, target=labels,
                                                                 probs=probs, jitter_vale=0.125,
                                                                 min_sigma=0.1, max_sigma=1.5,
                                                                 ignore_label=ignore_label)
    else:
        params["flip"] = random.random() < 0.5
        params["ColorJitter"] = random.random() < 0.8
        params["GaussianBlur"] = random.random() < 0.2
        params["Grayscale"] = random.random() < 0.0
        params["Solarize"] = random.random() < 0.0
        if random.random() < 0.80:
            scale = random.uniform(0.75, 1.75)
        else:
            scale = 1
        params["RandomScaleCrop"] = scale

        # Apply strong augmentations to unlabeled images
        image_aug, labels_aug, probs_aug = augmentationTransform(params,
                                                                 data=images, target=labels,
                                                                 probs=probs, jitter_vale=0.25,
                                                                 min_sigma=0.1, max_sigma=1.5,
                                                                 ignore_label=ignore_label)

    return image_aug, labels_aug, probs_aug, params

def main():
    print(config)

    cudnn.enabled = True
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True

    if pretraining == 'COCO': # depending the pretraining, normalize with bgr or rgb
        from utils.transformsgpu import normalize_bgr as normalize
    else:
        from utils.transformsgpu import normalize_rgb as normalize

    batch_size_unlabeled = int(batch_size / 2) # because of augmentation anchoring, 2 augmentations per sample
    batch_size_labeled = int(batch_size * 1)
    assert batch_size_unlabeled >= 2, "batch size should be higher than 2"
    assert batch_size_labeled >= 2, "batch size should be higher than 2"
    RAMP_UP_ITERS = 2000 # iterations until contrastive and self-training are taken into account

    # DATASETS / LOADERS
    if dataset == 'pascal_voc':
        data_loader = get_loader(dataset)
        data_path = get_data_path(dataset)
        train_dataset = data_loader(data_path, crop_size=input_size, scale=False, mirror=False, pretraining=pretraining)

    elif dataset == 'cityscapes':
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        if deeplabv2:
            data_aug = Compose([RandomCrop_city(input_size)])
        else: # for deeplabv3 original resolution
            data_aug = Compose([RandomCrop_city_highres(input_size)])
        train_dataset = data_loader(data_path, is_transform=True, augmentations=data_aug, img_size=input_size, pretraining=pretraining)

    train_dataset_size = len(train_dataset)
    print('dataset size: ', train_dataset_size)

    partial_size = labeled_samples
    print('Training on number of samples:', partial_size)

    # class weighting  taken unlabeled data into acount in an incremental fashion.
    class_weights_curr = ClassBalancing(labeled_iters=int(labeled_samples / batch_size_labeled),
                                                  unlabeled_iters=int(
                                                      (train_dataset_size - labeled_samples) / batch_size_unlabeled),
                                                  n_classes=num_classes)
    # Memory Bank
    feature_memory = FeatureMemory(num_samples=labeled_samples, dataset=dataset, memory_per_class=256, feature_size=256, n_classes=num_classes)

    # select the partition
    if split_id is not None:
        train_ids = pickle.load(open(split_id, 'rb'))
        print('loading train ids from {}'.format(split_id))
    else:
        train_ids = np.arange(train_dataset_size)
        np.random.shuffle(train_ids)

    # Samplers for labeled data
    train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=batch_size_labeled, sampler=train_sampler, num_workers=num_workers,
                                  pin_memory=True)
    trainloader_iter = iter(trainloader)

    # Samplers for unlabeled data
    train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:])
    trainloader_remain = data.DataLoader(train_dataset,
                                         batch_size=batch_size_unlabeled, sampler=train_remain_sampler,
                                         num_workers=num_workers, pin_memory=True)
    trainloader_remain_iter = iter(trainloader_remain)

    # supervised loss
    supervised_loss = CrossEntropy2d(ignore_label=ignore_label).cuda()

    ''' Deeplab model '''
    # Define network
    if deeplabv2:
        if pretraining == 'COCO': # coco and imagenet resnet architectures differ a little, just on how to do the stride
            from model.deeplabv2 import Res_Deeplab
        else: # imagenet pretrained (more modern modification)
            from model.deeplabv2_imagenet import Res_Deeplab

        # load pretrained parameters
        if pretraining == 'COCO':
            saved_state_dict = model_zoo.load_url('http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth') # COCO pretraining
        else:
            saved_state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth') # iamgenet pretrainning

    else:
        from model.deeplabv3 import Res_Deeplab50 as Res_Deeplab
        saved_state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth') # iamgenet pretrainning

    # create network
    model = Res_Deeplab(num_classes=num_classes)

    # Copy loaded parameters to model
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
    model.load_state_dict(new_params)


    # Optimizer for segmentation network
    learning_rate_object = Learning_Rate_Object(config['training']['learning_rate'])

    optimizer = torch.optim.SGD(model.optim_parameters(learning_rate_object),
                          lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    ema_model = create_ema_model(model, Res_Deeplab)
    ema_model.train()
    ema_model = ema_model.cuda()
    model.train()
    model = model.cuda()
    cudnn.benchmark = True

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + '/config.json', 'w') as handle:
        json.dump(config, handle, indent=4, sort_keys=False)
    pickle.dump(train_ids, open(os.path.join(checkpoint_dir, 'train_split.pkl'), 'wb'))

    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)

    epochs_since_start = 0
    start_iteration = 0
    best_mIoU = 0  # best metric while training

    # TRAINING
    for i_iter in range(start_iteration, num_iterations):
        model.train()  # set mode to training
        optimizer.zero_grad()

        loss_l_value = 0.
        adjust_learning_rate(optimizer, i_iter)

        ''' LABELED SAMPLES '''
        # Get batch
        try:
            batch = next(trainloader_iter)
            if batch[0].shape[0] != batch_size_labeled:
                batch = next(trainloader_iter)
        except:  # finish epoch, rebuild the iterator
            epochs_since_start = epochs_since_start + 1
            # print('Epochs since start: ',epochs_since_start)
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        images, labels, _, _, _ = batch
        images = images.cuda()
        labels = labels.cuda()

        ''' UNLABELED SAMPLES '''
        try:
            batch_remain = next(trainloader_remain_iter)
            if batch_remain[0].shape[0] != batch_size_unlabeled:
                batch_remain = next(trainloader_remain_iter)
        except:
            trainloader_remain_iter = iter(trainloader_remain)
            batch_remain = next(trainloader_remain_iter)

        # Unlabeled
        unlabeled_images, _, _, _, _ = batch_remain
        unlabeled_images = unlabeled_images.cuda()

        # Create pseudolabels
        with torch.no_grad():
            if use_teacher:
                logits_u_w, features_weak_unlabeled = ema_model(normalize(unlabeled_images, dataset), return_features=True)
            else:
                model.eval()
                logits_u_w, features_weak_unlabeled = model(normalize(unlabeled_images, dataset), return_features=True)
                model.train()

            logits_u_w = interp(logits_u_w).detach()  # prediction unlabeled
            softmax_u_w = torch.softmax(logits_u_w, dim=1)
            max_probs, pseudo_label = torch.max(softmax_u_w, dim=1)  # Get pseudolabels

        model.train()

        if dataset == 'cityscapes':
            class_weights_curr.add_frequencies(labels.cpu().numpy(), pseudo_label.cpu().numpy())


        images_aug, labels_aug, _, _ = augment_samples(images, labels, None, random.random()  < 0.2, batch_size_labeled, ignore_label, weak=True)

        '''
        UNLABELED DATA
        '''
        unlabeled_images_aug1, pseudo_label1, max_probs1, unlabeled_aug1_params = augment_samples(unlabeled_images,
                                                                                                  pseudo_label,
                                                                                                  max_probs,
                                                                                                  i_iter > RAMP_UP_ITERS and random.random() < 0.75,
                                                                                                  batch_size_unlabeled,
                                                                                                  ignore_label)


        unlabeled_images_aug2, pseudo_label2, max_probs2, unlabeled_aug2_params = augment_samples(unlabeled_images,
                                                                                                  pseudo_label,
                                                                                                  max_probs,
                                                                                                  i_iter > RAMP_UP_ITERS and random.random() < 0.75,
                                                                                                  batch_size_unlabeled,
                                                                                                  ignore_label)
        # concatenate two augmentations of unlabeled data
        joined_unlabeled = torch.cat((unlabeled_images_aug1, unlabeled_images_aug2), dim=0)
        joined_pseudolabels = torch.cat((pseudo_label1, pseudo_label2), dim=0)
        joined_maxprobs = torch.cat((max_probs1, max_probs2), dim=0)

        pred_joined_unlabeled, features_joined_unlabeled = model(normalize(joined_unlabeled, dataset), return_features=True)
        pred_joined_unlabeled = interp(pred_joined_unlabeled)

        # labeled data
        labeled_pred, labeled_features = model(normalize(images_aug, dataset), return_features=True)
        labeled_pred = interp(labeled_pred)

        # apply clas balance for cityspcaes dataset
        if dataset == 'cityscapes':
            class_weights = torch.from_numpy(
                class_weights_curr.get_weights(num_iterations, only_labeled=False)).cuda()
        else:
            class_weights = torch.from_numpy(np.ones((num_classes))).cuda()


        loss = 0

        # SUPERVISED SEGMENTATION
        labeled_loss = supervised_loss(labeled_pred, labels_aug, weight=class_weights.float())
        loss = loss + labeled_loss

        # SELF-SUPERVISED SEGMENTATION
        unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label, weight=class_weights.float()).cuda() #

        # Pseudo-label weighting
        pixelWiseWeight = sigmoid_ramp_up(i_iter, RAMP_UP_ITERS) * torch.ones(joined_maxprobs.shape).cuda()
        pixelWiseWeight = pixelWiseWeight * torch.pow(joined_maxprobs.detach(), 6)

        # Pseudo-label loss
        loss_ce_unlabeled = unlabeled_loss(pred_joined_unlabeled, joined_pseudolabels, pixelWiseWeight)

        loss = loss + loss_ce_unlabeled

        # entropy loss
        valid_mask = (joined_pseudolabels != ignore_label).unsqueeze(1)
        loss = loss + entropy_loss(torch.nn.functional.softmax(pred_joined_unlabeled, dim=1), valid_mask) * 0.01

        # CONTRASTIVE LEARNING
        if i_iter >  RAMP_UP_ITERS  - 1000:
            # Build Memory Bank 1000 iters before starting to do contrastive

            with torch.no_grad():
                # Get feature vectors from labeled images with EMA model
                if use_teacher:
                    labeled_pred_ema, labeled_features_ema = ema_model(normalize(images_aug, dataset), return_features=True)
                else:
                    model.eval()
                    labeled_pred_ema, labeled_features_ema = model(normalize(images_aug, dataset), return_features=True)
                    model.train()

                labeled_pred_ema = interp(labeled_pred_ema)
                probability_prediction_ema, label_prediction_ema = torch.max(torch.softmax(labeled_pred_ema, dim=1),dim=1)  # Get pseudolabels

            # Resize labels, predictions and probabilities,  to feature map resolution
            labels_down = nn.functional.interpolate(labels_aug.float().unsqueeze(1), size=(labeled_features_ema.shape[2], labeled_features_ema.shape[3]),
                                                    mode='nearest').squeeze(1)
            label_prediction_down = nn.functional.interpolate(label_prediction_ema.float().unsqueeze(1), size=(labeled_features_ema.shape[2], labeled_features_ema.shape[3]),
                                                    mode='nearest').squeeze(1)
            probability_prediction_down = nn.functional.interpolate(probability_prediction_ema.float().unsqueeze(1), size=(labeled_features_ema.shape[2], labeled_features_ema.shape[3]),
                                                    mode='nearest').squeeze(1)


            # get mask where the labeled predictions are correct and have a confidence higher than 0.95
            mask_prediction_correctly = ((label_prediction_down == labels_down).float() * (probability_prediction_down > 0.95).float()).bool()

            # Apply the filter mask to the features and its labels
            labeled_features_correct = labeled_features_ema.permute(0, 2, 3, 1)
            labels_down_correct = labels_down[mask_prediction_correctly]
            labeled_features_correct = labeled_features_correct[mask_prediction_correctly, ...]

            # get projected features
            with torch.no_grad():
                if use_teacher:
                    proj_labeled_features_correct = ema_model.projection_head(labeled_features_correct)
                else:
                    model.eval()
                    proj_labeled_features_correct = model.projection_head(labeled_features_correct)
                    model.train()
            # updated memory bank
            feature_memory.add_features_from_sample_learned(ema_model, proj_labeled_features_correct, labels_down_correct, batch_size_labeled)



        if i_iter > RAMP_UP_ITERS:
            '''
            CONTRASTIVE LEARNING ON LABELED DATA. Force features from labeled samples, to be similar to other features from the same class (which also leads to good predictions
            '''
            # mask features that do not have ignore label in the labels (zero-padding because of data augmentation like resize/crop)
            mask_prediction_correctly = (labels_down != ignore_label)

            labeled_features_all = labeled_features.permute(0, 2, 3, 1)
            labels_down_all = labels_down[mask_prediction_correctly]
            labeled_features_all = labeled_features_all[mask_prediction_correctly, ...]

            # get predicted features
            proj_labeled_features_all = model.projection_head(labeled_features_all)
            pred_labeled_features_all = model.prediction_head(proj_labeled_features_all)

            # Apply contrastive learning loss
            loss_contr_labeled = contrastive_class_to_class_learned_memory(model, pred_labeled_features_all, labels_down_all,
                                num_classes, feature_memory.memory)

            loss = loss + loss_contr_labeled * 0.1


            '''
            CONTRASTIVE LEARNING ON UNLABELED DATA. align unlabeled features to labeled features
            '''
            joined_pseudolabels_down = nn.functional.interpolate(joined_pseudolabels.float().unsqueeze(1),
                                                    size=(features_joined_unlabeled.shape[2], features_joined_unlabeled.shape[3]),
                                                    mode='nearest').squeeze(1)

            # mask features that do not have ignore label in the labels (zero-padding because of data augmentation like resize/crop)
            mask = (joined_pseudolabels_down != ignore_label)

            features_joined_unlabeled = features_joined_unlabeled.permute(0, 2, 3, 1)
            features_joined_unlabeled = features_joined_unlabeled[mask, ...]
            joined_pseudolabels_down = joined_pseudolabels_down[mask]

            # get predicted features
            proj_feat_unlabeled = model.projection_head(features_joined_unlabeled)
            pred_feat_unlabeled = model.prediction_head(proj_feat_unlabeled)

            # Apply contrastive learning loss
            loss_contr_unlabeled = contrastive_class_to_class_learned_memory(model, pred_feat_unlabeled, joined_pseudolabels_down,
                                num_classes, feature_memory.memory)

            loss = loss + loss_contr_unlabeled * 0.1


        loss_l_value += loss.item()

        # optimize
        loss.backward()
        optimizer.step()

        ema_model = update_ema_variables(ema_model=ema_model, model=model, alpha_teacher=0.996, iteration=i_iter)


        if i_iter % save_checkpoint_every == 0 and i_iter != 0:
            _save_checkpoint(i_iter, model, optimizer, config)

        if i_iter % val_per_iter == 0 and i_iter != 0:
            print('iter = {0:6d}/{1:6d}'.format(i_iter, num_iterations))

            model.eval()
            mIoU, eval_loss = evaluate(model, dataset, deeplabv2=deeplabv2, ignore_label=ignore_label, save_dir=checkpoint_dir, pretraining=pretraining)
            model.train()


            if mIoU > best_mIoU and save_best_model:
                best_mIoU = mIoU
                _save_checkpoint(i_iter, model, optimizer, config, save_best=True)

    _save_checkpoint(num_iterations, model, optimizer, config)

    model.eval()
    mIoU, val_loss = evaluate(model, dataset, deeplabv2=deeplabv2, ignore_label=ignore_label, save_dir=checkpoint_dir, pretraining=pretraining)

    if mIoU > best_mIoU and save_best_model:
        best_mIoU = mIoU
        _save_checkpoint(i_iter, model, optimizer, config, save_best=True)

    print('BEST MIOU')
    print(best_mIoU)

    end = timeit.default_timer()
    print('Total time: ' + str(end - start) + ' seconds')


if __name__ == '__main__':

    print('---------------------------------Starting---------------------------------')

    args = get_arguments()

    if args.resume:
        config = torch.load(args.resume)['config']
    else:
        config = json.load(open(args.config))

    model = config['model']
    dataset = config['dataset']

    if dataset == 'cityscapes':
        num_classes = 19
        if config['training']['data']['split_id_list'] == 0:
            split_id = './splits/city/split_0.pkl'
        elif config['training']['data']['split_id_list'] == 1:
            split_id = './splits/city/split_1.pkl'
        elif config['training']['data']['split_id_list'] == 2:
            split_id = './splits/city/split_2.pkl'
        else:
            split_id = None

    elif dataset == 'pascal_voc':
        num_classes = 21
        data_dir = './data/voc_dataset/'
        data_list_path = './data/voc_list/train_aug.txt'
        if config['training']['data']['split_id_list'] == 0:
            split_id = './splits/voc/split_0.pkl'
        elif config['training']['data']['split_id_list'] == 1:
            split_id = './splits/voc/split_1.pkl'
        elif config['training']['data']['split_id_list'] == 2:
            split_id = './splits/voc/split_2.pkl'
        else:
            split_id = None

    batch_size = config['training']['batch_size']
    num_iterations = config['training']['num_iterations']

    input_size_string = config['training']['data']['input_size']
    h, w = map(int, input_size_string.split(','))
    input_size = (h, w)

    ignore_label = config['ignore_label']  # 255 for PASCAL-VOC / 250 for Cityscapes

    learning_rate = config['training']['learning_rate']

    optimizer_type = config['training']['optimizer']
    lr_power = config['training']['lr_schedule_power']
    weight_decay = config['training']['weight_decay']
    momentum = config['training']['momentum']
    num_workers = config['training']['num_workers']
    random_seed = config['seed']
    pretraining = config['training']['pretraining']
    labeled_samples = config['training']['data']['labeled_samples']

    save_checkpoint_every = config['utils']['save_checkpoint_every']
    if args.resume:
        checkpoint_dir = os.path.join(*args.resume.split('/')[:-1]) + '_resume-'
    else:
        checkpoint_dir = os.path.join(config['utils']['checkpoint_dir'])
    log_dir = checkpoint_dir

    val_per_iter = config['utils']['val_per_iter']

    save_best_model = config['utils']['save_best_model']

    deeplabv2 = "2" in config['version']

    use_teacher = True # by default
    if 'use_teacher' in config['training']:
        use_teacher = config['training']['use_teacher']

    main()
