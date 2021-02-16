import argparse
import os
from data.augmentations import *
from utils.metric import ConfusionMatrix
from multiprocessing import Pool

from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from modeling.deeplab import *
from data.voc_dataset import VOCDataSet
from data import get_data_path, get_loader
import cv2
from utils.loss import CrossEntropy2d




def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SSL evaluation script")
    parser.add_argument("-m", "--model-path", type=str, default=None, required=True,
                        help="Model to evaluate")
    parser.add_argument("--gpu", type=int, default=(0,),
                        help="choose gpu device.")
    parser.add_argument("--save-output-images", action="store_true",
                        help="save output images")
    return parser.parse_args()



class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def get_label_vector(target, nclass):
    # target is a 3D Variable BxHxW, output is 2D BxnClass
    hist, _ = np.histogram(target, bins=nclass, range=(0, nclass - 1))
    vect = hist > 0
    vect_out = np.zeros((21, 1))
    for i in range(len(vect)):
        if vect[i] == True:
            vect_out[i] = 1
        else:
            vect_out[i] = 0

    return vect_out


def get_iou(confM, dataset, save_path=None):
    aveJ, j_list, M = confM.jaccard()

    if dataset == 'pascal_voc':
        classes = np.array(('background',  # always index 0
                            'aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair',
                            'cow', 'diningtable', 'dog', 'horse',
                            'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor'))
    elif dataset == 'cityscapes':
        classes = np.array(("road", "sidewalk",
                            "building", "wall", "fence", "pole",
                            "traffic_light", "traffic_sign", "vegetation",
                            "terrain", "sky", "person", "rider",
                            "car", "truck", "bus",
                            "train", "motorcycle", "bicycle"))

    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]))

    print('meanIOU: ' + str(aveJ) + '\n')

    return aveJ


def evaluate(model, dataset, ignore_label=250, save_dir=None, pretraining='COCO'):

    if pretraining == 'COCO':
        from utils.transformsgpu import normalize_bgr as normalize
    else:
        from utils.transformsgpu import normalize_rgb as normalize

    if dataset == 'pascal_voc':
        num_classes = 21
        input_size = (505, 505)
        data_loader = get_loader(dataset)
        data_path = get_data_path(dataset)
        test_dataset = data_loader(data_path, split="val", crop_size=input_size, scale=False, mirror=False, pretraining=pretraining)
        testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    elif dataset == 'cityscapes':
        num_classes = 19
        input_size = (512, 1024)  # like they are resize while training

        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        data_aug = Compose([Resize_city(input_size)])
        test_dataset = data_loader(data_path, img_size=input_size, is_transform=True, split='val',
                                   augmentations=data_aug, pretraining=pretraining)
        testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    print('Evaluating, found ' + str(len(testloader)) + ' images.')
    confM = ConfusionMatrix(num_classes)



    data_list = []

    total_loss = []

    for index, batch in enumerate(testloader):
        image, label, size, name, _ = batch
        class_names = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]
        colors = [  # [  0,   0,   0],
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ]
        with torch.no_grad():
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
            output, features = model(normalize(Variable(image).cuda(), dataset), return_features=True)
            output_classes = np.argmax(output[0, ...].cpu().numpy(), axis=0)
            features = features[0, ...].cpu().numpy()
            label = nn.functional.interpolate(label.float().unsqueeze(1),
                                                    size=(
                                                        output.shape[2], output.shape[3]),
                                                    mode='nearest').squeeze(1)
            output_classes = label[0, ...].cpu().numpy()
            output_classes = np.reshape(output_classes, (-1))
            features = np.swapaxes(np.reshape(features, (2048, -1)), 0, 1)
            print(output_classes.shape)
            print(features.shape)


            print(index)
            if index == 0:
                output_classes2 = output_classes
                features2 = features
            else:
                output_classes2 = np.concatenate((output_classes2, output_classes), axis=0)
                features2 = np.concatenate((features2, features), axis=0)

            if index == 5:

                from sklearn.decomposition import PCA
                pca = PCA(n_components=20)
                features2 = pca.fit_transform(features2)
                print(features2.shape)

                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, random_state=0)
                features2 = tsne.fit_transform(features2)
                print(features2.shape)
                print('aa')

                target_ids = range(19)
                from matplotlib import pyplot as plt
                plt.figure(figsize=(15, 8))
                for i, c, label in zip(target_ids, colors, class_names):
                    print(colors[i])
                    print(class_names[i])
                    plt.scatter(features2[output_classes2 == i, 0], features2[output_classes2 == i, 1], s=1,
                                c=(colors[i][0] / 255., colors[i][1] / 255., colors[i][2] / 255.))

                plt.legend()
                plt.show()
                import time
                time.sleep(1000)


def process_list_evaluation(confM, data_list):
    if len(data_list) > 0:
        f = confM.generateM
        pool = Pool(4)
        m_list = pool.map(f, data_list)
        pool.close()
        pool.join()
        pool.terminate()

        for m in m_list:
            confM.addM(m)



def main():
    """Create the model and start the evaluation process."""

    gpu0 = args.gpu


    deeplabv2 = "2" in config['version']

    if deeplabv2:
        if pretraining == 'COCO': # coco and iamgenet resnet architectures differ a little, just on how to do the stride
            from model.deeplabv2 import Res_Deeplab
        else: # imagenet pretrained (more modern modification)
            from model.deeplabv2_imagenet import Res_Deeplab

    else:
        from model.deeplabv3 import Res_Deeplab

    model = Res_Deeplab(num_classes=num_classes)

    checkpoint = torch.load(args.model_path)
    try:
        model.load_state_dict(checkpoint['model'])
    except:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
        model.load_state_dict(checkpoint['model'])


    model.cuda()
    model.eval()

    evaluate(model, dataset, ignore_label=ignore_label,   pretraining=pretraining)


if __name__ == '__main__':
    args = get_arguments()

    config = torch.load(args.model_path)['config']

    dataset = config['dataset']

    if dataset == 'cityscapes':
        num_classes = 19
        input_size = (512, 1024)
    elif dataset == 'pascal_voc':
        num_classes = 21

    ignore_label = config['ignore_label']

    pretraining = 'COCO'
    if pretraining == 'COCO':
        from utils.transformsgpu import normalize_bgr as normalize
    else:
        from utils.transformsgpu import normalize_rgb as normalize


    main()