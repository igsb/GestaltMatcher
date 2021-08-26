# GestaltMatcher - BlindDeepGestalt, ESHG 2021
IGSB implementation of DeepGestalt and a 'blind' version of it which supports the removal of learned biases.
This branch is implemented by Institute for Genomic Statistics and Bioinformatics (IGSB) at the University of Bonn. 
In this branch, we included two parts: the face cropper, and the model training. 
Due to the legal and copyright issue, the original photos for training and metadata are only available on demand. 
Please get in touch with us to access the data.

This work uses IGSB's GestaltMatcher base code, and newly created code for work described in https://arxiv.org/abs/1809.02169.
The paper introduces the Joint Learning Unlearning (JLU) algorithm to learn one feature of an image, while unlearning another.

In this branch we've chosen to use anonymized data from patients with Cornelia de Lange syndrome (CDLS) and 
Williams-Beuren syndrome (WBS).
The primary classification task of our model is to predict the syndrome, while the secondary task (to unlearn) is ethnicity.
The ethnicity has been split up into 'European' and 'Non-European' (due to the limited amount of available data). 

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
scikit-learn
```

## Data preparation
The data should be stored in `../data/GestaltMatcherDB_ESHG/`, it can be made available on request. 
We can provide the following two files:
* GMDB metadata
* gmdb_eshg_train_v1.tar.gz

```
cd ../data/GestaltMatcherDB_ESHG
tar -xzvf gmdb_eshg_train_v1.tar.gz
mv gmdb_eshg_train_v1 images
tar -xzvf GMDB_ESHG_metadata.tar.gz
mv GMDB_ESHG_metadata/* .
```

## Crop photos
In order to get the correct images, you have to run the `detect_pipe.py` from https://github.com/AlexanderHustinx/GestaltEngine-FaceCropper
More details are in the README of that repo. The face cropper requires the model "Resnet50_Final.pth".
Please remember to download the model from the repository mentioned above. 
If you don't have GPU, please use `--cpu` to run on cpu mode.

FaceCropper command to get all crops from data directory (used with main.py when selecting 'eshg' as dataset):
```
python detect_pipe.py --images_dir ../data/GestaltMatcherDB_ESHG/images/ --save_dir ../data/GestaltMatcherDB_ESHG/images_cropped/ --result_type crop
```


## Train models
The training of BlindDeepGestalt should be run on 'eshg' which uses basic augmentations on pre-cropped images.

You can run our BlindDeepGestalt model by training from scratch, using:
```
python main.py --dataset eshg --seed 11 --session 4 --num_classes 2 --model-type BlindDeepGestalt
```

By default the alpha-parameter (for the JLU algorithm) is set to 1.0, if you want to train with a different 
alpha value please use `--alpha <your value>`.

You may choose whatever seed and session you find useful.
`--seed 11` was used to obtain the results shown in the presentation, others have not been tested.

Using the argument `--use_tensorboard` allows you to track your models training and validation curves over time.

Training a model without GPU has not been tested, and is not recommended.

## Pretrained models
The most recent trained CASIA model weights can be downloaded here: <br />
https://drive.google.com/file/d/1yexBX2a9ny0fGDhOydXEGS_Qy_k_Pzzc

The trained BlindDeepGestalt model weights can be downloaded here: <br />
alpha=0.0: https://drive.google.com/file/d/17-di4PQQS62tIJt_rZpR79600GfOrTlm <br />
alpha=1.0: https://drive.google.com/file/d/1tVsnKw6cPxhpuDYAuNkClXeowymlcA8s 

The validation curves, using a perfectly balanced validation set, for both models are shown below. <br />
![Validation curves BlindDeepGestalt](https://github.com/igsb/GestaltMatcher/blob/eshg2021/graphs/validation_curves.jpg?raw=true) <br />
(Green) belongs to the BlindDeepGestalt with alpha=0.0, which shows that it is not unlearning the ethnicity, 
while (Orange) belongs to BlindDeepGestalt with alpha=1.0, showing it successfully unlearned ethnicity.

The expected directory for the model weights is `./saved_models/`

Please note that the CASIA model in this repository is required to train the DeepGestalt and BlindDeepGestalt models.


## Authors
Alexander Hustinx <br />

Tzung-Chien Hsieh (Corresponding) <br />
Email: thsieh@uni-bonn.de or la60312@gmail.com

## License
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc/4.0/)
