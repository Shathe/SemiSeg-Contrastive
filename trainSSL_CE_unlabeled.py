import argparse
import os
import sys
import random
import timeit
import datetime

import numpy as np
import pickle
import scipy.misc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform

from PIL import Image
from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted
from utils.loss import MSELoss2d

from utils import transformmasks
from utils import transformsgpu
from utils.helpers import colorize_mask
import utils.palette as palette

from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback

from modeling.deeplab import *

from data.voc_dataset import VOCDataSet

from data import get_loader, get_data_path
from data.augmentations import *
from tqdm import tqdm
from utils.transformsgpu import normalize

import PIL
from torchvision import transforms
import json
from torch.utils import tensorboard
from evaluateSSL import evaluate

import time
from utils.curriculum_class_balancing import CurriculumClassBalancing

start = timeit.default_timer()
start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--gpus", type=int, default=1,
                        help="choose number of gpu devices to use (default: 1)")
    parser.add_argument("-c", "--config", type=str, default='config.json',
                        help='Path to the config file (default: config.json)')
    parser.add_argument("-r", "--resume", type=str, default=None,
                        help='Path to the .pth file to resume from (default: None)')
    parser.add_argument("-n", "--name", type=str, default=None, required=True,
                        help='Name of the run (default: None)')
    parser.add_argument("--save-images", type=str, default=None,
                        help='Include to save images (default: None)')
    return parser.parse_args()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / float(max_iter)) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(learning_rate, i_iter, num_iterations, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def sigmoid_ramp_up(iter, max_iter):
    if iter >= max_iter:
        return 1
    else:
        return np.exp(- 5 * (1 - float(iter) / float(max_iter)) ** 2)


def sharpening(softmax, temperature=0.5):
    softmax_t = softmax.pow(1 / temperature)
    sum = torch.unsqueeze(softmax_t.sum(dim=1), dim=1)

    return softmax_t / sum


def focal_loss(prediction, gt, per_pixel_weights, ignore_label, gamma=0, alpha=1, do_softmax=True, from_one_hot=False,
               class_weight=None):
    assert not gt.requires_grad
    n, c, h, w = prediction.size()

    ignore_mask = (gt >= 0) * (gt != ignore_label)
    gt = gt[ignore_mask]
    per_pixel_weights = per_pixel_weights[ignore_mask]

    prediction = prediction.transpose(1, 2).transpose(2, 3).contiguous()
    prediction = prediction[ignore_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)

    if do_softmax:
        prediction = F.softmax(prediction, dim=1)
    prediction = torch.clamp(prediction, 1e-20, 1)

    if not from_one_hot:
        gt = F.one_hot(gt, c)  # cuidado ignore label

    assert prediction.shape == gt.shape

    loss = - torch.log(prediction) * gt

    prediction = torch.clamp(prediction, 1e-20, 1 - 1e-4)  # potential problems if gamma < 0
    weight_focal = torch.pow((1 - prediction), gamma) * gt

    loss = loss * weight_focal

    loss = loss * per_pixel_weights.unsqueeze(1).expand(-1, c)
    loss = loss.mean(dim=0)

    if class_weight is not None:
        loss = loss * class_weight
        loss = loss.sum()
    else:
        loss = loss.sum()

    return alpha * loss


def pseudolabel_weighting(probabilities, mul=20, prob_half=0.85):
    return 1 / (1 + torch.exp(-(probabilities - prob_half) * mul))


def augmentationTransform(parameters, data=None, target=None, probs=None, jitter_vale=0.2, min_sigma=0.2, max_sigma=2.,
                          scale=2., ignore_label=255):
    assert ((data is not None) or (target is not None))
    if "Mix" in parameters:
        data, target, probs = transformsgpu.mix(mask=parameters["Mix"], data=data, target=target, probs=probs)

    if "flip" in parameters:
        data, target, probs = transformsgpu.flip(flip=parameters["flip"], data=data, target=target, probs=probs)
    if "RandomScaleCrop" in parameters:
        data, target, probs = transformsgpu.random_scale_crop(scale_crop=parameters["RandomScaleCrop"], data=data,
                                                              target=target, probs=probs, scale=scale,
                                                              ignore_label=ignore_label)
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


def getWeakInverseTransformParameters(parameters):
    return parameters


def getStrongInverseTransformParameters(parameters):
    return parameters


class Learning_Rate_Object(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


def save_image(image, epoch, id, palette):
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
                transforms.ToPILImage()])

            image = restore_transform(image)
            # image = PIL.Image.fromarray(np.array(image)[:, :, ::-1])  # BGR->RGB
            image.save(os.path.join('../visualiseImages/', str(epoch) + id + '.png'))
        else:
            mask = image.numpy()
            colorized_mask = colorize_mask(mask, palette)
            colorized_mask.save(os.path.join('../visualiseImages/', str(epoch) + id + '.png'))


def _save_checkpoint(iteration, model, optimizer, config, save_best=False, overwrite=True):
    checkpoint = {
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    if len(gpus) > 1:
        checkpoint['model'] = model.module.state_dict()
    else:
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


def _resume_checkpoint(resume_path, model, optimizer):
    print(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)

    # Load last run info, the model params, the optimizer and the loggers
    iteration = checkpoint['iteration'] + 1
    print('Starting at iteration: ' + str(iteration))

    if len(gpus) > 1:
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    return iteration, model, optimizer


def CBC_thresholding(max_probs, pseudo_label, num_classes, ignore_label, percentage_of_labels=1.,
                     rescale_confidence=False, downsample_ratio=1):
    probabilities = [np.array([], dtype=np.float32) for _ in range(num_classes)]
    probs = max_probs.flatten()[:: downsample_ratio]
    labels = pseudo_label.flatten()[:: downsample_ratio]

    for j in range(num_classes):
        probabilities[j] = np.concatenate((probabilities[j],
                                           probs[labels == j].cpu().numpy()))
    kc = []
    for j in range(num_classes):
        if len(probabilities[j]) > 0:
            probabilities[j].sort()
            number_of_labels = max(int(len(probabilities[j]) * percentage_of_labels), 1)
            kc.append(probabilities[j][-number_of_labels])
        else:
            kc.append(1.)
    del probabilities  # Better be safe than...

    for c in range(num_classes):
        mask_class = pseudo_label == c
        mask_probs = max_probs < kc[c]
        mask_remove = mask_class * mask_probs
        pseudo_label[mask_remove] = ignore_label
        max_probs[mask_remove] = 0.
        if rescale_confidence and max_probs[mask_class].shape[0] != 0:
            max_probs[mask_class] = max_probs[mask_class] / max_probs[mask_class].max()

    return max_probs, pseudo_label


def main():
    print(config)
    cudnn.enabled = True
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True

    batch_size_unlabeled = int(batch_size / 2)
    RAMP_UP_ITERS = 10000
    # DATASETS
    if dataset == 'pascal_voc':
        data_loader = get_loader(dataset)
        data_path = get_data_path(dataset)
        train_dataset = data_loader(data_path, crop_size=input_size, scale=False, mirror=False)

    elif dataset == 'cityscapes':
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        data_aug = Compose(
            [RandomCrop_city(input_size)])  # from 1024x2048 to resize 512x1024 to crop input_size (512x512)
        train_dataset = data_loader(data_path, is_transform=True, augmentations=data_aug, img_size=input_size)

    train_dataset_size = len(train_dataset)
    print('dataset size: ', train_dataset_size)

    partial_size = labeled_samples
    print('Training on number of samples:', partial_size)

    class_weights_curr = CurriculumClassBalancing(ramp_up=RAMP_UP_ITERS,
                                                  labeled_samples=int(labeled_samples / batch_size),
                                                  unlabeled_samples=int(
                                                      (train_dataset_size - labeled_samples) / batch_size_unlabeled),
                                                  n_classes=num_classes)

    # select the partition
    if split_id is not None:
        train_ids = pickle.load(open(split_id, 'rb'))
        print('loading train ids from {}'.format(split_id))
    else:
        train_ids = np.arange(train_dataset_size)
        np.random.shuffle(train_ids)

    train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
                                  pin_memory=True)
    trainloader_iter = iter(trainloader)

    if train_unlabeled:
        train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:])
        trainloader_remain = data.DataLoader(train_dataset,
                                             batch_size=batch_size_unlabeled, sampler=train_remain_sampler,
                                             num_workers=num_workers, pin_memory=True)
        trainloader_remain_iter = iter(trainloader_remain)

    # LOSSES
    if len(gpus) > 1:
        unlabeled_loss = torch.nn.DataParallel(CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label),
                                               device_ids=gpus).cuda()
        supervised_loss = torch.nn.DataParallel(CrossEntropy2d(ignore_label=ignore_label), device_ids=gpus).cuda()
    else:
        unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
        supervised_loss = CrossEntropy2d(ignore_label=ignore_label).cuda()

    ''' Deeplab model '''
    # Define network
    model = DeepLab(num_classes=num_classes, backbone="resnet", output_stride=16, sync_bn=use_sync_batchnorm,
                    freeze_bn=False, v2=deeplabv2)

    train_params = [{'params': model.get_1x_lr_params(), 'lr': learning_rate},
                    {'params': model.get_10x_lr_params(), 'lr': learning_rate * 10}]

    # Define Optimizer
    optimizer = torch.optim.SGD(train_params, momentum=momentum, weight_decay=weight_decay)

    if len(gpus) > 1:
        if use_sync_batchnorm:
            model = convert_model(model)
            model = DataParallelWithCallback(model, device_ids=gpus)
        else:
            model = torch.nn.DataParallel(model, device_ids=gpus)
    model.train()
    model.cuda()
    cudnn.benchmark = True

    # checkpoint = torch.load('/home/snowflake/Escritorio/Semi-Sup/saved/DeepLab_copy3/checkpoint-iter35000.pth')
    # model.load_state_dict(checkpoint['model'])

    if args.resume:
        start_iteration, model, optimizer = _resume_checkpoint(args.resume, model, optimizer)

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
        loss_u_value = 0.
        adjust_learning_rate(optimizer, i_iter)

        ''' LABELED SAMPLES '''
        # Get batch
        try:
            batch = next(trainloader_iter)
            if batch[0].shape[0] != batch_size:
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
            model.eval()
            logits_u_w = interp(model(normalize(unlabeled_images.detach())).detach())  # prediction unlabeled
            softmax_u_w = torch.softmax(logits_u_w, dim=1)
            max_probs, pseudo_label = torch.max(softmax_u_w, dim=1)  # Get pseudolabels
            # free memory
            del softmax_u_w
            del logits_u_w

        model.train()

        class_weights_curr.add_frequencies(labels.cpu().numpy(), pseudo_label.cpu().numpy(),
                                           None)  # max_probs.cpu().numpy()

        # Weak augmentations.
        labeled_ce_aug_params = {}
        labeled_ce_aug_params["flip"] = random.random() < 0.5

        # Apply weak augmentations to labeled images
        images, labels, _ = augmentationTransform(labeled_ce_aug_params, data=images, target=labels,
                                                  ignore_label=ignore_label, probs=None)

        class_weights = torch.from_numpy(
            class_weights_curr.get_weights(num_iterations, reduction_freqs=np.sum, only_labeled=False)).cuda()

        # Get supervised loss
        pred = interp(model(normalize(images)))
        loss = supervised_loss(pred, labels, weight=class_weights.float())

        '''
        UNLABELED DATA
        '''
        # ClassMix: Get mask for image A
        for image_i in range(batch_size_unlabeled):  # for each image
            classes = torch.unique(pseudo_label[image_i])  # get unique classes in pseudolabel A
            nclasses = classes.shape[0]
            # pick half of the classes randomly
            classes = (classes[torch.Tensor(
                np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2), replace=False)).long()]).cuda()

            if dataset == 'pascal_voc':  # if voc dataaset, remove class 0, background
                classes = classes[classes != 0]

            # acumulate masks
            if image_i == 0:
                MixMask = transformmasks.generate_class_mask(pseudo_label[image_i], classes).unsqueeze(0).cuda()
            else:
                MixMask = torch.cat(
                    (MixMask, transformmasks.generate_class_mask(pseudo_label[image_i], classes).unsqueeze(0).cuda()))

        '''
        CROSS ENTROPY FOR UNLABELED USING PSEUDOLABELS
        Once you have the speudolabel, perform strong augmetnation to force the netowrk to yield lower confidence scores for pushing them up
        '''
        do_classmix = i_iter > RAMP_UP_ITERS and random.random() > 0.5 # only after rampup perfrom classmix
        if do_classmix:
            unlabeled_aug1_params = {"Mix": MixMask}
        else:
            unlabeled_aug1_params = {}

        # same sa BYOL, and, classmix on one
        unlabeled_aug1_params["flip"] = random.random() < 0.5
        unlabeled_aug1_params["ColorJitter"] = random.random() < 0.8
        unlabeled_aug1_params["GaussianBlur"] = random.random() <= 1.
        unlabeled_aug1_params["RandomScaleCrop"] = random.random() <= 1.
        unlabeled_aug1_params["Grayscale"] = random.random() < 0.2
        # unlabeled_aug1_params["Solarize"] = random.random() < 0.1

        # Apply strong augmentations to unlabeled images
        unlabeled_images_aug1, pseudo_label1, max_probs1 = augmentationTransform(unlabeled_aug1_params,
                                                                                 data=unlabeled_images,
                                                                                 target=pseudo_label, probs=max_probs,
                                                                                 jitter_vale=random.randint(0,
                                                                                                            40) / 100.,
                                                                                 min_sigma=0.1, max_sigma=2., scale=2.5,
                                                                                 ignore_label=ignore_label)

        unlabeled_aug2_params = {}
        unlabeled_aug2_params["flip"] = random.random() < 0.5
        unlabeled_aug2_params["ColorJitter"] = random.random() < 0.8
        unlabeled_aug2_params["GaussianBlur"] = random.random() < 0.1
        unlabeled_aug2_params["RandomScaleCrop"] = random.random() <= 1.
        unlabeled_aug2_params["Grayscale"] = random.random() < 0.2
        unlabeled_aug2_params["Solarize"] = random.random() < 0.2

        # Apply strong augmentations to unlabeled images
        unlabeled_images_aug2, pseudo_label2, max_probs2 = augmentationTransform(unlabeled_aug2_params,
                                                                                 data=unlabeled_images,
                                                                                 target=pseudo_label, probs=max_probs,
                                                                                 jitter_vale=random.randint(0,
                                                                                                            40) / 100.,
                                                                                 min_sigma=0.1, max_sigma=2, scale=2.5,
                                                                                 ignore_label=ignore_label)

        joined_unlabeled = torch.cat((unlabeled_images_aug1, unlabeled_images_aug2), dim=0)
        pred_joined_unlabeled = interp(model(normalize(joined_unlabeled)))
        perd_unlabeled_1, perd_unlabeled_2 = torch.split(pred_joined_unlabeled, batch_size_unlabeled, dim=0)


        '''
        Cross entropy loss using pseudolabels
        '''
        # Pseudo-label weighting
        # TODO: watch out for giving the correct max_probs that fits the perd_unlabeled_1 and pseudo_label1
        pixelWiseWeight = sigmoid_ramp_up(i_iter, RAMP_UP_ITERS) * torch.ones(max_probs1.shape).cuda()
        pixelWiseWeight = pixelWiseWeight * torch.pow(max_probs1.detach(), 9)

        # Pseudo-label loss
        unlabeled_loss = torch.nn.DataParallel(
            CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label, weight=class_weights.float()),
            device_ids=gpus).cuda()
        loss_ce_unlabeled = consistency_weight * unlabeled_loss(perd_unlabeled_1, pseudo_label1,
                                                                pixelWiseWeight)  # change self.weight (comprobar que da difernete antes y depsues de coambarlo)

        loss = loss + loss_ce_unlabeled

        # TODO: watch out for giving the correct max_probs that fits the perd_unlabeled_2 and pseudo_label2
        pixelWiseWeight = sigmoid_ramp_up(i_iter, RAMP_UP_ITERS) * torch.ones(max_probs2.shape).cuda()
        pixelWiseWeight = pixelWiseWeight * torch.pow(max_probs2.detach(), 9)

        # Pseudo-label loss
        loss_ce_unlabeled = consistency_weight * unlabeled_loss(perd_unlabeled_2, pseudo_label2,
                                                                pixelWiseWeight)  # change self.weight (comprobar que da difernete antes y depsues de coambarlo)

        loss = loss + loss_ce_unlabeled

        # import cv2
        # image = unlabeled_images[0, ...].cpu().numpy().copy()
        # label = pseudo_label[0, ...].cpu().numpy().copy()
        # image = np.swapaxes(image, 0, 1)
        # image = np.swapaxes(image, 2, 1)
        # image = image[:, :, ::-1]
        #
        #
        # cv2.imshow('img', image.astype(np.uint8))
        # cv2.imshow('label', label.astype(np.uint8)*10)

        # AUGMENTATION

        # image = unlabeled_images_aug_ce[0, ...].cpu().numpy()
        # label = pseudo_label[0, ...].cpu().numpy().copy()
        # image = np.swapaxes(image, 0, 1)
        # image = np.swapaxes(image, 2, 1)
        # image = image[:, :, ::-1]
        # cv2.imshow('img2', image.astype(np.uint8))
        # cv2.imshow('label2', label.astype(np.uint8)*10)
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #

        if len(gpus) > 1:
            loss = loss.mean()
            loss_l_value += loss.mean().item()
        else:
            loss_l_value += loss.item()

        # optimize
        loss.backward()
        optimizer.step()

        # print('iter = {0:6d}/{1:6d}, loss_l = {2:.3f}'.format(i_iter, num_iterations, loss_l_value))

        if i_iter % save_checkpoint_every == 0 and i_iter != 0:
            _save_checkpoint(i_iter, model, optimizer, config)

        if i_iter % val_per_iter == 0 and i_iter != 0:
            print('iter = {0:6d}/{1:6d}'.format(i_iter, num_iterations))

            model.eval()
            mIoU, eval_loss = evaluate(model, dataset, ignore_label=ignore_label, save_dir=checkpoint_dir)
            model.train()

            if mIoU > best_mIoU and save_best_model:
                best_mIoU = mIoU
                _save_checkpoint(i_iter, model, optimizer, config, save_best=True)

    _save_checkpoint(num_iterations, model, optimizer, config)

    model.eval()
    mIoU, val_loss = evaluate(model, dataset, ignore_label=ignore_label, save_dir=checkpoint_dir)

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
    lr_schedule = config['training']['lr_schedule']
    lr_power = config['training']['lr_schedule_power']
    weight_decay = config['training']['weight_decay']
    momentum = config['training']['momentum']
    num_workers = config['training']['num_workers']
    use_sync_batchnorm = config['training']['use_sync_batchnorm']
    random_seed = config['seed']

    labeled_samples = config['training']['data']['labeled_samples']

    # unlabeled CONFIGURATIONS
    train_unlabeled = config['training']['unlabeled']['train_unlabeled']
    pixel_weight = config['training']['unlabeled']['pixel_weight']
    consistency_loss = config['training']['unlabeled']['consistency_loss']
    consistency_weight = config['training']['unlabeled']['consistency_weight']

    save_checkpoint_every = config['utils']['save_checkpoint_every']
    if args.resume:
        checkpoint_dir = os.path.join(*args.resume.split('/')[:-1]) + '_resume-'
    else:
        checkpoint_dir = os.path.join(config['utils']['checkpoint_dir'])
    log_dir = checkpoint_dir

    val_per_iter = config['utils']['val_per_iter']
    log_per_iter = config['utils']['log_per_iter']

    save_best_model = config['utils']['save_best_model']

    gpus = (0, 1, 2, 3)[:args.gpus]
    deeplabv2 = "2" in config['version']

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

    main()
