import argparse
import os
import timeit
import datetime
import cv2
import pickle
from contrastive_losses import *
import torch.backends.cudnn as cudnn
from torch.utils import data
from torch.utils import data, model_zoo

from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted

from utils import transformmasks
from utils import transformsgpu
from utils.helpers import colorize_mask

from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback

from modeling.deeplab import *

from data import get_loader, get_data_path
from data.augmentations import *
from utils.transformsgpu import normalize

from torchvision import transforms
import json
from evaluateSSL import evaluate
import time
from utils.curriculum_class_balancing import CurriculumClassBalancing
from utils.feature_memory import *

start = timeit.default_timer()
start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')


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


def augmentationTransform(parameters, data=None, target=None, probs=None, jitter_vale=0.4, min_sigma=0.2, max_sigma=2., ignore_label=255):
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

def create_ema_model(model):
    if deeplabv2:
        from model.deeplabv2 import Res_Deeplab
    else:
        from model.deeplabv3 import Res_Deeplab

    ema_model = Res_Deeplab(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    if len(gpus)>1:
        if use_sync_batchnorm:
            ema_model = convert_model(ema_model)
            ema_model = DataParallelWithCallback(ema_model, device_ids=gpus)
        else:
            ema_model = torch.nn.DataParallel(ema_model, device_ids=gpus)
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    if len(gpus)>1:
        for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model





def augment_samples(images, labels, probs, do_classmix, batch_size, ignore_label):
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
    params["ColorJitter"] = random.random() < 0.80
    params["GaussianBlur"] = random.random() < 0.2
    params["Grayscale"] = random.random() < 0.0
    params["Solarize"] = random.random() < 0.0
    if random.random() < 0.75:
        scale = random.uniform(0.75, 1.75)
    else:
        scale = 1
    params["RandomScaleCrop"] = scale

    # Apply strong augmentations to unlabeled images
    image_aug, labels_aug, probs_aug = augmentationTransform(params,
                                                             data=images, target=labels,
                                                             probs=probs, jitter_vale=0.25,
                                                             min_sigma=0.1, max_sigma=1.25,
                                                             ignore_label=ignore_label)

    return image_aug, labels_aug, probs_aug, params


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
    if random.random() < 0.33:
        scale = random.uniform(0.85, 1.5)
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
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    supervised_unlabeled_loss = True
    supervised_labeled_loss = True
    contrastive_labeled_loss = False

    batch_size_unlabeled = int(batch_size / 2)
    batch_size_labeled = int(batch_size * 1 )

    RAMP_UP_ITERS = 2000

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
                                                  labeled_samples=int(labeled_samples / batch_size_labeled),
                                                  unlabeled_samples=int(
                                                      (train_dataset_size - labeled_samples) / batch_size_unlabeled),
                                                  n_classes=num_classes)

    feature_memory = FeatureMemory(num_samples=labeled_samples, dataset=dataset, memory_per_class=2048, feature_size=256, n_classes=num_classes)

    # select the partition
    if split_id is not None:
        train_ids = pickle.load(open(split_id, 'rb'))
        print('loading train ids from {}'.format(split_id))
    else:
        train_ids = np.arange(train_dataset_size)
        np.random.shuffle(train_ids)

    train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=batch_size_labeled, sampler=train_sampler, num_workers=num_workers,
                                  pin_memory=True)
    trainloader_iter = iter(trainloader)

    if train_unlabeled:
        train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:])
        trainloader_remain = data.DataLoader(train_dataset,
                                             batch_size=batch_size_unlabeled, sampler=train_remain_sampler,
                                             num_workers=num_workers, pin_memory=True)
        trainloader_remain_iter = iter(trainloader_remain)

    # LOSSES
    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    supervised_loss = CrossEntropy2d(ignore_label=ignore_label).cuda()

    ''' Deeplab model '''
    # Define network

    if deeplabv2:
        from model.deeplabv2 import Res_Deeplab
    else:
        from model.deeplabv3 import Res_Deeplab

    # create network
    model = Res_Deeplab(num_classes=num_classes)

    # load pretrained parameters
    saved_state_dict = model_zoo.load_url('http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth') # COCO pretraining
    # saved_state_dict = model_zoo.load_url(''https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'') # iamgenet pretrainning

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

    ema_model = create_ema_model(model)
    ema_model.train()
    ema_model = ema_model.cuda()

    if len(gpus) > 1:
        if use_sync_batchnorm:
            model = convert_model(model)
            model = DataParallelWithCallback(model, device_ids=gpus)
        else:
            model = torch.nn.DataParallel(model, device_ids=gpus)
    model.train()
    model.cuda()
    cudnn.benchmark = True

    # checkpoint = torch.load('/home/snowflake/checkpoint-iter50000.pth')
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
        a = time.time()

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
            logits_u_w, features_weak_unlabeled = ema_model(normalize(unlabeled_images, dataset), return_features=True)
            logits_u_w = interp(logits_u_w).detach()  # prediction unlabeled
            softmax_u_w = torch.softmax(logits_u_w, dim=1)
            max_probs, pseudo_label = torch.max(softmax_u_w, dim=1)  # Get pseudolabels

        model.train()

        class_weights_curr.add_frequencies(labels.cpu().numpy(), pseudo_label.cpu().numpy(), None)

        images, labels, _, _ = augment_samples_weak(images, labels, None, random.random()  < 0.15, batch_size_labeled, ignore_label)

        '''
        UNLABELED DATA
        '''

        '''
        CROSS ENTROPY FOR UNLABELED USING PSEUDOLABELS
        Once you have the speudolabel, perform strong augmetnation to force the netowrk to yield lower confidence scores for pushing them up
        '''

        do_classmix = i_iter > RAMP_UP_ITERS and random.random() < 0.5  # only after rampup perfrom classmix
        unlabeled_images_aug1, pseudo_label1, max_probs1, unlabeled_aug1_params = augment_samples(unlabeled_images,
                                                                                                  pseudo_label,
                                                                                                  max_probs,
                                                                                                  do_classmix,
                                                                                                  batch_size_unlabeled,
                                                                                                  ignore_label)

        do_classmix = i_iter > RAMP_UP_ITERS and random.random() < 0.5  # only after rampup perfrom classmix

        unlabeled_images_aug2, pseudo_label2, max_probs2, unlabeled_aug2_params = augment_samples(unlabeled_images,
                                                                                                  pseudo_label,
                                                                                                  max_probs,
                                                                                                  do_classmix,
                                                                                                  batch_size_unlabeled,
                                                                                                  ignore_label)


        joined_unlabeled = torch.cat((unlabeled_images_aug1, unlabeled_images_aug2), dim=0)
        joined_pseudolabels = torch.cat((pseudo_label1, pseudo_label2), dim=0)
        joined_maxprobs = torch.cat((max_probs1, max_probs2), dim=0)

        pred_joined_unlabeled, features_joined_unlabeled = model(normalize(joined_unlabeled, dataset), return_features=True)
        pred_joined_unlabeled = interp(pred_joined_unlabeled)


        joined_labeled = images
        joined_labels = labels
        labeled_pred, labeled_features = model(normalize(joined_labeled, dataset), return_features=True)
        labeled_pred = interp(labeled_pred)

        class_weights = torch.from_numpy(
            class_weights_curr.get_weights(num_iterations, reduction_freqs=np.sum, only_labeled=False)).cuda()

        loss = 0
        if supervised_labeled_loss:
            labeled_loss = supervised_loss(labeled_pred, joined_labels, weight=class_weights.float()) # weight=class_weights.float()
            loss = loss + labeled_loss

        if supervised_unlabeled_loss:
            '''
            Cross entropy loss using pseudolabels. 
            '''
            unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label, weight=class_weights.float()).cuda() #

            # Pseudo-label weighting
            pixelWiseWeight = sigmoid_ramp_up(i_iter, RAMP_UP_ITERS) * torch.ones(joined_maxprobs.shape).cuda()
            pixelWiseWeight = pixelWiseWeight * torch.pow(joined_maxprobs.detach(), 9)

            # Pseudo-label loss
            loss_ce_unlabeled = unlabeled_loss(pred_joined_unlabeled, joined_pseudolabels, pixelWiseWeight)

            loss = loss + loss_ce_unlabeled

            # entropy loss
            valid_mask = (joined_pseudolabels != ignore_label).unsqueeze(1)
            loss = loss + entropy_loss(torch.nn.functional.softmax(pred_joined_unlabeled, dim=1), valid_mask) * 0.01

        if contrastive_labeled_loss:

            # this is sueprvised contrastive learning
            #if RAMP_UP_ITERS  - 1000:
            if i_iter > RAMP_UP_ITERS  - 1000:  # RAMP_UP_ITERS  - 1000:
                # TODO: DEJAS ESTO Y LO DE ABAJO DE EMA PARA PROTOTYPES?
                # Create prototypes from labeled images with EMA model
                with torch.no_grad():
                    labeled_pred, labeled_features = ema_model(normalize(joined_labeled, dataset), return_features=True)
                    labeled_pred = interp(labeled_pred)
                    _, label_prediction = torch.max(torch.softmax(labeled_pred, dim=1), dim=1)# Get pseudolabels

                '''
                We are going to pick only faetures from labeled samples that are predicting correctly the label
                As the feature resolutoin could be different from the output resolution (common in semseg), 
                we do it in the feature reoslution (Downsampling the labels and predictions) due to computational restrictions
                
                Doing this per-class instead to per-class-distribution, gives us the benefits of:
                - quicker filtering of good feautres (otherwise you need a threhosld for the accuracy (you comapre class dsitributions and they will hardly never be the same)
                - for saving in the  memory, for saving different ones, its easiest to save N per class instead of computing the entrpy between the difernt distributions to se which oen to sve)
                - for filtering in the memory its easiest
                - to save which class/class distribution is the feature you are saving, this way its just hte class. 
                    Otherwise, there would be the question about, to save the labeled class distribution or the predicted class dsitribution?
                '''
                labels_down = nn.functional.interpolate(joined_labels.float().unsqueeze(1),
                                                        size=(labeled_features.shape[2], labeled_features.shape[3]),
                                                        mode='nearest').squeeze(1)
                label_prediction_down = nn.functional.interpolate(label_prediction.float().unsqueeze(1),
                                                        size=(labeled_features.shape[2], labeled_features.shape[3]),
                                                        mode='nearest').squeeze(1)

                # get mask where the labeled predictions are correct
                mask_prediction_correctly = (label_prediction_down == labels_down)

                labeled_features_correct = labeled_features.permute(0, 2, 3, 1)
                labels_down_correct = labels_down[mask_prediction_correctly]
                labeled_features_correct = labeled_features_correct[mask_prediction_correctly, ...]

                # get projected features
                with torch.no_grad():
                    proj_labeled_features_correct = ema_model.projection_head(labeled_features_correct)

                feature_memory.add_features_from_sample(proj_labeled_features_correct, labels_down_correct, batch_size_labeled)


                # # get label distribution from labels
                # class_dist = F.one_hot(labels, 255)
                # class_dist = class_dist.permute(0, 3, 1, 2)
                # class_dist = torch.nn.functional.avg_pool2d(class_dist.float(),
                #                 kernel_size=int( labeled_pred.shape[2] / labeled_features.shape[2]))
                # # rull out ignore label
                # class_dist = class_dist[:, :num_classes, :, :]
                # # renormalize distribution
                # class_dist = class_dist / torch.sum(class_dist, dim=1).unsqueeze(1)

                # take only features which lead to accurate predictions
                # threhsold > 0.5 accuracy. select only good features


            # TODO: this is sueprvised contrastive learning
            #if i_iter > RAMP_UP_ITERS:
            if i_iter > RAMP_UP_ITERS:  # RAMP_UP_ITERS:
                '''
                LABELED TO LABELED. Force features from laeled samples, to be similar to other features from the same class (which also leads to good predictions)
                
                '''
                # First, get the predicted probability of the expected labeled
                label_prediction_probs = torch.softmax(labeled_pred, dim=1)
                joined_labels_aux = joined_labels.clone()
                joined_labels_aux[joined_labels==ignore_label] = num_classes
                one_hot_labels = F.one_hot(joined_labels_aux, num_classes + 1).permute(0, 3, 1, 2)
                correct_labeled_probs = label_prediction_probs * one_hot_labels[:, :num_classes, :, :]
                correct_labeled_probs = correct_labeled_probs.sum(dim=1)

                labeled_pred_probs_down = nn.functional.interpolate(correct_labeled_probs.unsqueeze(1),
                                                        size=(labeled_features.shape[2], labeled_features.shape[3]),
                                                        mode='nearest').squeeze(1)



                # now we can take all. as they are not the prototypes, here we are gonan force these features to be similar as the correct ones
                mask_prediction_correctly = (labels_down != ignore_label)

                labeled_features_all = labeled_features.permute(0, 2, 3, 1)
                labels_down_all = labels_down[mask_prediction_correctly]
                labeled_prediction_probs_all = labeled_pred_probs_down[mask_prediction_correctly]
                labeled_features_all = labeled_features_all[mask_prediction_correctly, ...]

                # get prediction features
                proj_labeled_features_all = model.projection_head(labeled_features_all)
                pred_labeled_features_all = model.prediction_head(proj_labeled_features_all)

                loss_contr_labeled = contrastive_class_to_class(pred_labeled_features_all, labels_down_all, labeled_prediction_probs_all,
                                    batch_size_labeled, num_classes, feature_memory.memory, None)

                loss = loss + loss_contr_labeled

                '''
                UNLABELED TO LABELED
                '''
                # First, get the predicted probability of the expected labeled
                unlabel_prediction_probs = torch.softmax(pred_joined_unlabeled, dim=1)
                joined_pseudolabels_aux = joined_pseudolabels.clone()
                joined_pseudolabels_aux[joined_pseudolabels==ignore_label] = num_classes
                one_hot_pseudolabels = F.one_hot(joined_pseudolabels_aux, num_classes + 1).permute(0, 3, 1, 2)
                correct_unlabeled_probs = unlabel_prediction_probs * one_hot_pseudolabels[:, :num_classes, :, :]
                correct_unlabeled_probs = correct_unlabeled_probs.sum(dim=1)

                unlabeled_prediction_probs_down = nn.functional.interpolate(correct_unlabeled_probs.float().unsqueeze(1),
                                                        size=(features_joined_unlabeled.shape[2], features_joined_unlabeled.shape[3]),
                                                        mode='nearest').squeeze(1)
                joined_pseudolabels_down = nn.functional.interpolate(joined_pseudolabels.float().unsqueeze(1),
                                                        size=(features_joined_unlabeled.shape[2], features_joined_unlabeled.shape[3]),
                                                        mode='nearest').squeeze(1)
                joined_maxprobs_down = nn.functional.interpolate(joined_maxprobs.float().unsqueeze(1),
                                                        size=(features_joined_unlabeled.shape[2], features_joined_unlabeled.shape[3]),
                                                        mode='nearest').squeeze(1)

                # take out the features from black pixels from zooms out and augmetnations (ignore labels on pseduoalebl)
                mask = (joined_pseudolabels_down != ignore_label)

                features_joined_unlabeled = features_joined_unlabeled.permute(0, 2, 3, 1)
                features_joined_unlabeled = features_joined_unlabeled[mask, ...]
                joined_pseudolabels_down = joined_pseudolabels_down[mask]
                unlabeled_prediction_probs_down = unlabeled_prediction_probs_down[mask]
                joined_maxprobs_down = joined_maxprobs_down[mask]

                # get projected features
                proj_feat_unlabeled = model.projection_head(features_joined_unlabeled)
                pred_feat_unlabeled = model.prediction_head(proj_feat_unlabeled)

                loss_contr_unlabeled = contrastive_class_to_class(pred_feat_unlabeled, joined_pseudolabels_down, unlabeled_prediction_probs_down,
                                    batch_size_unlabeled, num_classes, feature_memory.memory, joined_maxprobs_down)

                loss = loss + loss_contr_unlabeled

                '''
                Pasos:
                - sacar features igual que en labeled. Elegir M por clase y dependiendo del abtch (auqeu ahora segurament elegir mas elementos).
                - mirar implementacion del otro donde multilpicaba tanto por el suyo como por otros y esoapra cada pixel
                
                Posibles ablations:
                
                - minimizar erro de.. todos vs solo los M de menor valor.. con varias opciones. mientas, piensa cosa automaticas como segun venico o cosas
                - dar peso segun pseudolabels confidence o solo usar pseudoalebls con mayor confianza que 0.95
                
                - comprar o con N priermos o, csimeplemten comprar con todos los que sean menores qeu X o regla del segundo vecino
    
                 - meter las features seleccionadas mejor en vez de random 
             
                  que la memoria sea solo con los buenos accuracies pero luego eso no tenerlo en cuanta en los 
                  que fuerzas y samples para alinear y forzar mismas features buenas además de la misma clase
                '''

        # print(time.time() - a)

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

        ema_model = update_ema_variables(ema_model=ema_model, model=model, alpha_teacher=0.99,
                                         iteration=i_iter)

        # print('iter = {0:6d}/{1:6d}, loss_l = {2:.3f}'.format(i_iter, num_iterations, loss_l_value))

        if i_iter % save_checkpoint_every == 0 and i_iter != 0:
            _save_checkpoint(i_iter, model, optimizer, config)

        if i_iter % val_per_iter == 0 and i_iter != 0:
            print('iter = {0:6d}/{1:6d}'.format(i_iter, num_iterations))

            model.eval()
            mIoU, eval_loss = evaluate(model, dataset, ignore_label=ignore_label, save_dir=checkpoint_dir)
            model.train()
            if supervised_labeled_loss:
                print('last labeled loss')
                print(labeled_loss)
            if contrastive_labeled_loss and i_iter > RAMP_UP_ITERS:
                print('last loss_pix_to_pix loss')
                print(loss_contr_unlabeled)
                print(loss_contr_unlabeled)
            print('need to rebalance?')

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

    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

    main()
