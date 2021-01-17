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
from utils.transformsgpu import normalize

from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback

from modeling.deeplab import *

from data.voc_dataset import VOCDataSet

from data import get_loader, get_data_path
from data.augmentations import *
from tqdm import tqdm

import PIL
from torchvision import transforms
import json
from torch.utils import tensorboard
from evaluateSSL3 import evaluate

import time

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
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(learning_rate, i_iter, num_iterations, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def sigmoid_ramp_up(iter, max_iter):
    if iter >= max_iter:
        return 1
    else:
        return np.exp(- 5 * (1 - iter / max_iter) ** 2)





def augmentationTransform(parameters, data=None, target=None, probs=None, jitter_vale=0.4, min_sigma=0.2, max_sigma=2.,
                          scale=(0.5, 2.5), ignore_label=255):
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



def getWeakInverseTransformParameters(parameters):
    return parameters

def getStrongInverseTransformParameters(parameters):
    return parameters


class Learning_Rate_Object(object):
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

def save_image(image, epoch, id, palette):
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
            transforms.ToPILImage()])

            image = restore_transform(image)
            #image = PIL.Image.fromarray(np.array(image)[:, :, ::-1])  # BGR->RGB
            image.save(os.path.join('../visualiseImages/', str(epoch)+ id + '.png'))
        else:
            mask = image.numpy()
            colorized_mask = colorize_mask(mask, palette)
            colorized_mask.save(os.path.join('../visualiseImages/', str(epoch)+ id + '.png'))

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


def augment_samples_weak(images, labels, probs, do_classmix, batch_size, ignore_label):
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

    # similar as BYOL, plus, classmix
    params["flip"] = random.random() < 0.5
    params["ColorJitter"] = random.random() < 0.20
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

    return image_aug, labels_aug, probs_aug, params



def main():
    print(config)
    cudnn.enabled = True

    # DATASETS
    if dataset == 'pascal_voc':
        data_loader = get_loader(dataset)
        data_path = get_data_path(dataset)
        train_dataset = data_loader(data_path, crop_size=input_size, scale=False, mirror=False)

    elif dataset == 'cityscapes':
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        data_aug = Compose([RandomCrop_city(input_size)]) # from 1024x2048 to resize 512x1024 to crop input_size (512x512)
        train_dataset = data_loader(data_path, is_transform=True, augmentations=data_aug, img_size=input_size)


    train_dataset_size = len(train_dataset)
    print ('dataset size: ', train_dataset_size)

    partial_size = labeled_samples
    print('Training on number of samples:', partial_size)

    # select the partition
    if split_id is not None:
        train_ids = pickle.load(open(split_id, 'rb'))
        print('loading train ids from {}'.format(split_id))
    else:
        print('new seed')
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.backends.cudnn.deterministic = True
        train_ids = np.arange(train_dataset_size)
        np.random.shuffle(train_ids)

    train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
    trainloader = data.DataLoader(train_dataset,
                    batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    trainloader_iter = iter(trainloader)

    if train_unlabeled:
        train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:])
        trainloader_remain = data.DataLoader(train_dataset,
                    batch_size=batch_size, sampler=train_remain_sampler, num_workers=num_workers, pin_memory=True)
        trainloader_remain_iter = iter(trainloader_remain)

    # LOSSES
    if len(gpus) > 1:
        unlabeled_loss = torch.nn.DataParallel(CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label), device_ids=gpus).cuda()
        supervised_loss = torch.nn.DataParallel(CrossEntropy2d(ignore_label=ignore_label), device_ids=gpus).cuda()
    else:
        unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
        supervised_loss = CrossEntropy2d(ignore_label=ignore_label).cuda()


    if deeplabv2:
        from model.deeplabv2 import Res_Deeplab
    else:
        from model.deeplabv3 import Res_Deeplab

    # create network
    model = Res_Deeplab(num_classes=num_classes)

    # load pretrained parameters
    saved_state_dict = model_zoo.load_url('http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth')

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




    if len(gpus)>1:
        if use_sync_batchnorm:
            model = convert_model(model)
            model = DataParallelWithCallback(model, device_ids=gpus)
        else:
            model = torch.nn.DataParallel(model, device_ids=gpus)
    model.train()
    model.cuda()
    cudnn.benchmark = True



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
    best_mIoU = 0 # best metric while training

    model.eval()
    mIoU, eval_loss = evaluate(model, dataset, ignore_label=ignore_label, save_dir=checkpoint_dir)

    # TRAINING
    for i_iter in range(start_iteration, num_iterations):
        model.train() # set mode to training
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
        except: # finish epoch, rebuild the iterator
            epochs_since_start = epochs_since_start + 1
            # print('Epochs since start: ',epochs_since_start)
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        images, labels, _, name, _ = batch
        images = images.cuda()
        labels = labels.cuda()

        # Apply weak augmentations to labeled images
        images, labels, _, _ = augment_samples_weak(images, labels, None, random.random()  < 0.20, batch_size, ignore_label)




        pred = interp(model(normalize(images, dataset)))

        # Get supervised loss
        loss_labeled = supervised_loss(pred, labels) # supervised loss


        loss = loss_labeled


        if len(gpus) > 1:
            loss = loss.mean()
            loss_l_value += loss_labeled.mean().item()
        else:
            loss_l_value += loss_labeled.item()

        # optimize
        loss.backward()
        optimizer.step()

        # print('iter = {0:6d}/{1:6d}, loss_l = {2:.3f}, loss_u = {3:.3f}'.format(i_iter, num_iterations, loss_l_value, loss_u_value))

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
    print('Total time: ' + str(end-start) + ' seconds')


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

    ignore_label = config['ignore_label'] # 255 for PASCAL-VOC / 250 for Cityscapes

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

    #unlabeled CONFIGURATIONS
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

    gpus = (0,1,2,3)[:args.gpus]
    deeplabv2 = "2" in config['version']

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    main()
