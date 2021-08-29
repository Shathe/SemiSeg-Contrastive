This repository provides the official code for replicating experiments from the paper:
**Semi-Supervised Semantic Segmentation with Pixel-Level Contrastive Learning from a Class-wise Memory Bank** which as been accepted as an **oral** paper in the IEEE International Conference on Computer Vision (**ICCV**) 2021.

This code is based on [ClassMix code](https://github.com/WilhelmT/ClassMix)

# Semi-Supervised Semantic Segmentation with Pixel-Level Contrastive Learning from a Class-wise Memory Bank

## Prerequisites
*  CUDA/CUDNN 
*  Python3
*  Packages found in requirements.txt


## Contact
If any question, please either open a github issue or contact via email to: <inigoalonsoruiz@gmail.com>

## Datasets

Create a folder outsite the code folder:
```
mkdir ../data/
```

### Cityscapes
```
mkdir ../data/CityScapes/
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

Here there are some examples for replicating the experiments from the paper.
Implementation details are specified in the paper (section 4.2) **any modification** could potentially affect to the final result.


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

For replicating paper experiments, just execute the training of the specific set-up to replicate. We already provide all the configuration files used in the paper. For modifying them and a detail description of all the parameters in the configuration files, check this example:

#### Configuration File Description
```
{
  "model": "DeepLab", # Network architecture. Options: Deeplab
  "version": "2", # Version of the network architecture. Options: {2, 3} for deeplabv2 and deeplabv3+
  "dataset": "cityscapes", # Dataset to use. Options: {"cityscapes", "pascal"}

  "training": { 
    "batch_size": 5, # Batch size to use. Options: any integer
    "num_workers": 3, # Number of cpu workers (threads) to use for laoding the dataset. Options: any integer
    "optimizer": "SGD", # Optimizer to use. Options: {"SGD"}
    "momentum": 0.9, # momentum for SGD optimizer, Options: any float 
    "num_iterations": 100000, # Number of iterations to train. Options: any integer
    "learning_rate": 2e-4, # Learning rate. Options: any float
    "lr_schedule": "Poly", # decay scheduler for the learning rate. Options: {"Poly"}
    "lr_schedule_power": 0.9, # Power value for the Poly scheduler. Options: any float
    "pretraining": "COCO", # Pretraining to use. Options: {"COCO", "imagenet"}
    "weight_decay": 5e-4, # Weight decay. Options: any float
    "use_teacher_train": true, # Whether to use the teacher network to generate pseudolabels. Use student otherwise. Options: boolean. 
    "save_teacher_test": false, # Whether to save the teacher network as the model for testing. Use student otherwise. Options: boolean. 
    
    "data": {
      "split_id_list": 0, # Data splits to use. Options: {0, 1, 2} for pre-computed splits. N >2 for random splits
      "labeled_samples": 744, # Number of labeled samples to use for supervised learning. The rest will be use without labels. Options: any integer
      "input_size": "512,512" # Image crop size  Options: any integer tuple
    }

  },
  "seed": 5555, # seed for randomization. Options: any integer
  "ignore_label": 250, # ignore label value. Options: any integer

  "utils": {
    "save_checkpoint_every": 10000,  # The model will be saved every this number of iterations. Options: any integer
    "checkpoint_dir": "../saved/DeepLab", # Path to save the models. Options: any path
    "val_per_iter": 1000, # The model will be evaluated every this number of iterations. Options: any integer
    "save_best_model": true # Whether to use teacher model for generating the psuedolabels. The student model wil obe used otherwise. Options: boolean
  }
}
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
python3 trainSSL_domain_adaptation_targetCity.py --config ./configs/configSSL_city_1_30_split0_imagenet.json 
```

### Evaluation
The training code will evaluate the training model every some specific number of iterations (modify the parameter val_per_iter in the configuration file).

Best evaluated model will be printed at the end of the training.

For every training, several weights will be saved under the path specified in the parameter checkpoint_dir of the configuration file.

One model every save_checkpoint_every (see configuration file) will be saved, plus the best evaluated model.

So, the model has trained we can already know the performance.

For a later evaluation, just execute the next command specifying the model to evaluate in the model-path argument:
```
python3 evaluateSSL.py --model-path ../saved/DeepLab/best.pth
```


### Citation
If you find this work useful, please consider citing:
```BibTeX
@inproceedings{alonso2021semi,
  title={Semi-Supervised Semantic Segmentation with Pixel-Level Contrastive Learning from a Class-wise Memory Bank},
  author={Alonso, I{\~n}igo and Sabater, Alberto and Ferstl, David and Montesano, Luis and Murillo, Ana C},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2021}
}
```
