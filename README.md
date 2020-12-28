Code used for the results in the paper  ["ClassMix: Segmentation-Based Data Augmentation for Semi-Supervised Learning"](https://arxiv.org/abs/2007.07936)
# Getting started
## Prerequisite
*  CUDA/CUDNN 
*  Python3
*  Packages found in requirements.txt

## Datasets

### Cityscapes
Download the dataset from the Cityscapes dataset server([Link](https://www.cityscapes-dataset.com/)). Download the files named 'gtFine_trainvaltest.zip', 'leftImg8bit_trainvaltest.zip' and extract in ../data/Cityscapes/

### Pascal VOC 2012
Download the dataset from ([Link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)). Download the file 'training/validation data' under 'Development kit' and extract in ../data/VOC2012/

Arguments related to running the script are specified from terminal and include; number of gpus to use (if >1 torch.nn.DataParalell is used), path to configuration file (see below), path to .pth file if resuming training, name of the experiment, and whether to save images during training. More details can be found in the relevant scripts.

Arguments related to the algoritms are specified in the configuration files. These include model, data, hyperparameters related to the training, and what methods to apply on unlabeled data. A full description is provided further below.

### Example of training a model with semi-supervised learning on CityScapes with 12.5% the labels on a single gpu

python3 trainSSL.py --config ./configs/configSSL.json --name SSL

### Example of resuming training of a model with semi-supervised learning

python3 trainSSL.py --resume-path *checkpoint.pth*


# other
CUDA_LAUNCH_BLOCKING=1 nohup python3 trainSupervised.py --config ./configs/configSSL_city_1_30_split0_supervised.json --name SSL --gpus 1  > output_nohup/experiment_onlysup_deeplav2_noaug.txt &
CUDA_LAUNCH_BLOCKING=1 nohup sh trainSSL_sup.sh  > output_nohup/experiment_.txt &
