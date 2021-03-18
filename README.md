Official code for replicating experiments from the paper:
**Improving Semi-Supervised Semantic Segmentation with Class-Wise and Pixel-Level Contrastive Learning**

This code is based on [ClassMix code](https://github.com/WilhelmT/ClassMix)

# Getting started
## Prerequisites
*  CUDA/CUDNN 
*  Python3
*  Packages found in requirements.txt

## Datasets

Create a folder outsite the code folder:
```
mkdir ../data/
```

### Cityscapes
```
mkdir ../data/Cityscapes/
```
Download the dataset from ([Link](https://www.cityscapes-dataset.com/)).

Download the files named 'gtFine_trainvaltest.zip', 'leftImg8bit_trainvaltest.zip' and extract in ../data/Cityscapes/

### Pascal VOC 2012
```
mkdir ../data/VOC2012/
```
Download the dataset from ([Link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)). 

Download the file 'training/validation data' under 'Development kit' and extract in ../data/VOC2012/

### GTA5 
```
mkdir ../data/GTA5/
```
Download the dataset from ([Link](https://download.visinf.tu-darmstadt.de/data/from_games/)).
Unzip all the datasets parts to create an structure like this:
```
../data/GTA5/images/val/*.png
../data/GTA5/images/train/*.png
../data/GTA5/labels/val/*.png
../data/GTA5/labels/train/*.png
```

Then, reformat the label images from colored images to training ids.
For that, execute this:
```
python3 utils/translate_labels.py
```

## Experiments
### Semi-Supervised
Search here for the desired configuration:
```
ls ./configs/
```
For example, for this configuration:
* Dataset: CityScapes
* % of labels:  1/30  
* Pretrain: COCO
* Split: 0
* Network: Deeplabv2

Execute:

```
python3 trainSSL.py --config ./configs/configSSL_city_1_30_split0_COCO.json 
```

Another example, for this configuration:
* Dataset: CityScapes
* % of labels:  1/30  
* Pretrain: imagenet
* Split: 0
* Network: Deeplabv3+

Execute:

```
python3 trainSSL.py --config ./configs/configSSL_city_1_30_split0_v3.json 
```


For example, for this configuration:
* Dataset: PASCAL VOC
* % of labels:  1/50  
* Pretrain: COCO
* Split: 0

Execute:

```
python3 trainSSL.py --config ./configs/configSSL_pascal_1_50_split0_COCO.json 
```


### Memory Restrictions

All experiments have been run in an NVIDIA Tesla V100. To try to fit the training in a smaller GPU, try to follow this tips:

* Reduce batch_size from the configuration file 
* Reduce input_size from the configuration file 
* Instead of using trainSSL.py use trainSSL_less_memory.py which optimized labeled and unlabeled data separate steps.



For example, for this configuration:
* Dataset: PASCAL VOC
* % of labels:  1/50  
* Pretrain: COCO
* Split: 0
* Batch size: 8
* Crop size: 256x256
Execute:

```
python3 trainSSL_less_memory.py --config ./configs/configSSL_pascal_1_50_split2_COCO_reduced.json 
```


### Semi-Supervised Domain Adaptation

Experiments for domain adaptation from GTA5 dataset to Cityscapes.

For example, for configuration:
* % of labels:  1/30  
* Pretrain: Imagenet
* Split: 0

Execute:
```
python3 trainSSL.py --config ./configs/configSSL_city_1_30_split0_imagenet.json 
```

### Configuration File Description
```
{
  "model": "DeepLab", # Options: Deeplab
  "version": "2", # Options: {2, 3} for deeplabv2 and deeplabv3+
  "dataset": "cityscapes", # Options: {"cityscapes", "pascal"}

  "training": {
    "batch_size": 5, # Options: any integer
    "num_workers": 3, # Options: any integer
    "optimizer": "SGD", # Options: {"SGD"}
    "momentum": 0.9, # momentum for SGD optimizer, Options: any float 
    "num_iterations": 80000, # Options: any integer
    "learning_rate": 2e-4, # Options: any float
    "lr_schedule": "Poly", # Options: {"Poly"}
    "lr_schedule_power": 0.9, # power value for the Poly scheduler. Options: any float
    "pretraining": "COCO", # Options: {"COCO", "imagenet"}
    "weight_decay": 5e-4, # Options: any float
    "use_teacher": true, # Whether to use the teacher network to generate pseudolabels. Use student otherwise. Options: boolean. 
    
    "data": {
      "split_id_list": 0, # Options: {0,1,2} for pre-computed splits. N >2 for random splits
      "labeled_samples": 744, # Options: any integer
      "input_size": "512,512" # Options: any integer tuple
    }

  },
  "seed": 5555, # seed for randomization. Options: any integer
  "ignore_label": 250, # ignore label value. Options: any integer

  "utils": {
    "save_checkpoint_every": 10000,  # Options: any integer
    "checkpoint_dir": "../saved/DeepLab", # Options: any path
    "val_per_iter": 1000, # Options: any integer
    "save_best_model": true # Options: boolean
  }
}
```

### Citation
Soon