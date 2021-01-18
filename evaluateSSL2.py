import argparse
import os
from data.augmentations import *
from utils.metric import ConfusionMatrix
from multiprocessing import Pool

from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from utils.transformsgpu import normalize_inverse as normalize
from modeling.deeplab import *
from data.voc_dataset import VOCDataSet
from data import get_data_path, get_loader

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
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]) + '\n')
            f.write('meanIOU: ' + str(aveJ) + '\n')
    return aveJ


def evaluate(model, dataset, ignore_label=250, save_dir=None):



    if dataset == 'pascal_voc':
        num_classes = 21
        input_size = (505, 505)
        from data.voc_dataset2 import VOCDataSet
        data_loader = VOCDataSet
        data_path = get_data_path(dataset)
        test_dataset = data_loader(data_path, split="val", crop_size=input_size, scale=False, mirror=False)
        testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    elif dataset == 'cityscapes':
        num_classes = 19
        input_size = (512, 1024)  # like they are resize while training

        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        data_aug = Compose([Resize_city(input_size)])
        test_dataset = data_loader(data_path, img_size=input_size, is_transform=True, split='val',
                                   augmentations=data_aug)
        testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    print('Evaluating, found ' + str(len(testloader)) + ' images.')
    confM = ConfusionMatrix(num_classes)



    data_list = []

    total_loss = []

    for index, batch in enumerate(testloader):
        image, label, size, name, _ = batch

        with torch.no_grad():
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
            output = model(normalize(Variable(image).cuda(), dataset))
            output = interp(output)

            label_cuda = Variable(label.long()).cuda()
            criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()
            loss = criterion(output, label_cuda)
            total_loss.append(loss.item())

            output = output.cpu().data[0].numpy()
            gt = np.asarray(label[0].numpy(), dtype=np.int)

            output = np.asarray(np.argmax(output, axis=0), dtype=np.int)
            data_list.append((np.reshape(gt, (-1)), np.reshape(output, (-1))))

        if (index + 1) % 100 == 0:
            print('%d processed' % (index + 1))
            process_list_evaluation(confM, data_list)
            data_list = []

    process_list_evaluation(confM, data_list)

    if save_dir:
        filename = os.path.join(save_dir, 'result.txt')
    else:
        filename = None
    mIoU = get_iou(confM, dataset, filename)
    loss = np.mean(total_loss)
    return mIoU, loss


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

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    deeplabv2 = "2" in config['version']
    model = DeepLab(num_classes=num_classes, backbone='resnet',  sync_bn=False,
                    freeze_bn=False, v2=deeplabv2)
    checkpoint = torch.load(args.model_path)
    try:
        model.load_state_dict(checkpoint['model'])
    except:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
        model.load_state_dict(checkpoint['model'])


    model.cuda()
    model.eval()

    evaluate(model, dataset, ignore_label=ignore_label, save_dir=save_dir)


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
    save_dir = os.path.join(*args.model_path.split('/')[:-1])

    main()