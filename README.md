# GestaltMatcher
Unofficial implementation of GestaltMatcher as described in https://www.medrxiv.org/content/10.1101/2020.12.28.20248193v2.

## Environment

Please use python version 3.8, and the package listed in requirements.txt.

```
python3 -m venv env_gm
source env_gm/Scripts/activate
pip install -r requirements.txt
```

If you would like to train and evaluate with GPU, please remember to install cuda in your system.
If you don't have GPU, please choose the CPU option in the following section.

Follow these instructions (https://developer.nvidia.com/cuda-downloads ) to properly install CUDA.
Follow the necessary instructions (https://pytorch.org/get-started/locally/ ) to properly install PyTorch, you might still need additional dependencies (e.g. Numpy).
Using the following command should work for most using the `conda` virtual env.
```conda install pytorch torchvision cudatoolkit=10.2 -c pytorch```

If any problems occur when installing the packages in `requirements.txt`, the most important packages are:
```
numpy
pandas
pytorch=1.9.0
torchvision=0.10.0
tensorboard
opencv
matplotlib
```

## Data preparation
The data should be stored in `../data/GestaltMatcherDB/`, it can be downloaded from http://gestaltmatcher.org on request.
Depending on which model you want to train, either `images_cropped/` for the normal augmentations or `images_rot/` for the extensive ones.

In order to get the correct images, you have to run the `detect_pipe.py` from https://github.com/AlexanderHustinx/GestaltEngine-FaceCropper
More details are in the README of that repo. The face cropper requires the model "Resnet50_Final.pth".
Please remember to download the model from the repository mentioned above.  

FaceCropper command to get all crops from data directory (used with main.py when selecting 'gmdb' as dataset AND predict.py):
```
python detect_pipe.py --images_dir ../data/GestaltMatcherDB/images/ --save_dir ../data/GestaltMatcherDB/images_cropped/ --result_type crop
```

FaceCropper command to get all aligned faces and their coords from data directory (used with main.py when selecting 'gmdb_aug' as dataset):
```
python detect_pipe.py --images_dir ../data/GestaltMatcherDB/images/ --save_dir ../data/GestaltMatcherDB/images_rot/ --result_type coords
```

Do note that the crops are required to get the encodings with `predict.py`.

## Train models
The training of GestaltMatcher can be run on either 'gmdb' which uses basic augmentations on pre-cropped images, or 'gmdb_aug' which uses more extensive augmentations on the images. The latter reduces the models tendancy to overfit due to it creating more variation in the training data.

To reproduce our Gestalt Matcher model listed in the table by training from scratch, use:
```
python main.py --dataset gmdb_aug --seed 11 --session 2 --num_classes 139 --model-type DeepGestalt
```

In order to run them, the augementation in `./lib/augmentations` from https://github.com/AlexanderHustinx/LandmarkAugmentations have been used.

You may choose whatever seed and session you find useful.
`--seed 11` was used to obtain these results, others have not been tested.

Using the argument `--use_tensorboard` allows you to track your models training and validation curves over time.

Training a model without GPU has not been tested, and is not recommended.

## Pretrained models
The most recent trained CASIA model weights can be downloaded here:
https://drive.google.com/file/d/1yexBX2a9ny0fGDhOydXEGS_Qy_k_Pzzc

The most recent trained DeepGestalt model weights (with extensive augmentations) can be downloaded here:
https://drive.google.com/file/d/1SeHEvBqTYeb2uR65UT9Di5DmP1DEKb4B

The expected directory for the model weights is `./saved_models/`

Please note that the CASIA model in this repository is the same as Enc-healthy in the GestaltMatcher paper.

## Encode photos and evaluate models
With `python predict.py` you will evaluate all images in `--data_dir`. 
By default, it will use the recent DeepGestalt model with a batch size of 1 on the GPU ("saved_models/s2_gmdb_adam_DeepGestalt_e310_ReLU_BN_bs280.pt").
It will load all images in the `--data_dir`, which by default is set to `../data/GestaltMatcherDB/images_cropped`

When retraining a model with different parameters, make sure to check if the `--num_classes` is the same.
The classes for GMDB model is 139 which is the default setting. The classes for CASIA model is 10575.
The face encodings with will be saved to `encodings.csv` for DeepGestalt, and to `healthy_encoding.csv` for FaceRecogNet.

For the machine without GPU, please use `--no-cuda`.

The following two commands will generate encodings.csv (Enc-GMDB) and healthy_encodings.csv (Enc-CASIA).

```
# Encode images with GMDB model and output in encodings.csv
python predict.py --data_dir ../data/GestaltMatcherDB/images_cropped --no-cuda --model-type DeepGestalt --save_encodings

# Encode images with CAISA model and output in healthy_encodings.csv
python predict.py --data_dir ../data/GestaltMatcherDB/images_cropped --no-cuda --model-type FaceRecogNet --save_encodings --num_classes 10575
```


Using these encodings as input for <<INSERT SCRIPT TZUNG>> will allow you to obtain the results listed in the table.

## Results
The tables below hold the results from the original paper, and the reproductions provided in this repo (Enc-healthy, Enc-GMDB, Enc-GMDB softmax).

For the GMDB-frequent test set, using a gallery of 3428 images of 139 syndromes:

| Model | Top-1 | Top-5 | Top-10 | Top-30 |
|:-|:-|:-|:-|:-|
| Enc-GMDB softmax<br>ours | 29.98%<br>25.10% | 48.31%<br>46.67% | 66.30%<br>61.61% | 81.71%<br>78.46% |
| Enc-GMDB<br>ours | 21.86%<br>20.64% | 40.09%<br>41.50% | 53.59%<br>54.10% | 74.28%<br>71.67% |
| Enc-healthy<br>ours| 17.04%<br>16.29% | 33.26%<br>36.00% | 44.03%<br>42.52% | 63.46%<br>64.31% |

For the GMDB-rare test set, using a gallery of 369.2 images of 118 syndromes:
Model | Top-1 | Top-5 | Top-10 | Top-30 |  
|---|---|---|---|---|
| Enc-GMDB&emsp;&emsp;&emsp;&ensp;&nbsp;<br>ours | 18.51%<br>15.78% | 37.50%<br>34.68% | 47.36%<br>46.19% | 71.93%<br>69.24% |
| Enc-healthy<br>ours | 14.85%<br>12.79% | 30.53%<br>29.06% | 40.43%<br>40.19% | 61.65%<br>61.50% |
  
For the GMDB-frequent test set, using a gallery of 3812 images of 257 syndromes:
| Model | Top-1 | Top-5 | Top-10 | Top-30 |
|---|---|---|---|---|
| Enc-GMDB&emsp;&emsp;&emsp;&ensp;&nbsp;<br>ours | 20.98%<br>18.34% | 38.25%<br>40.19% | 51.05%<br>52.21% | 71.37%<br>68.17% |
| Enc-healthy<br>ours | 15.14%<br>15.89% | 31.14%<br>35.94% | 42.20%<br>41.00% | 62.48%<br>62.62% |
  
For the GMDB-frequent test set, using a gallery of 3428 images of 139 syndromes:
| Model | Top-1 | Top-5 | Top-10 | Top-30 |
|---|---|---|---|---|
| Enc-GMDB&emsp;&emsp;&emsp;&ensp;&nbsp;<br>ours | 8.47%&ensp;<br>7.77% | 18.33%<br>14.86% | 23.19%<br>20.42% | 37.62%<br>36.12% |
| Enc-healthy<br>ours | 7.10%<br>6.75% | 14.36%<br>13.08% | 19.34%<br>17.15% | 30.77%<br>28.29% |

