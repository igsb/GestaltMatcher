# GestaltMatcher

## Data preparation
The data should be stored in `../data/GestaltMatcherDB/`, it can be downloaded from www.gestalt-matcher.org on request.
Depending on which model you want to train, either `images_cropped/` for the normal augmentations or `images_rot/` for the extensive ones.

In order to get the correct images, you have to run the `detect_pipe.py` from https://github.com/AlexanderHustinx/GestaltEngine-FaceCropper
More details are in the README of that repo.

To run the face cropper you can use the following command: 
`python detect_pipe.py --images_dir ../data/GestaltMatcherDB/images/ --save_dir ../data/GestaltMatcherDB/images_cropped/` 
if you plan on using the extensive augmentation add `--result-type coords`, and use 
`--save_dir ../data/GestaltMatcherDB/images_rot/`.

## Train models
To reproduce our Gestalt Matcher model listed in the table by training from scratch, use:
`python main.py --dataset gmdb_aug --seed 11 --session 2 --num_classes 139 --model-type DeepGestalt`

In order to run them, make sure to git-pull the `augmentations` from https://github.com/AlexanderHustinx/LandmarkAugmentations into the `lib`-directory

You may choose whatever seed and session you find useful.
`--seed 11` was used to obtain these results, others have not been tested.

## Pretrained models
The most recent trained CASIA model weights can be downloaded here:
https://drive.google.com/file/d/1yexBX2a9ny0fGDhOydXEGS_Qy_k_Pzzc

The most recent trained DeepGestalt model weights (with extensive augmentations) can be downloaded here:
https://drive.google.com/file/d/1SeHEvBqTYeb2uR65UT9Di5DmP1DEKb4B

The expected directory for the model weights is `./saved_models/`

## Evaluate models
With `python predict.py` you will evaluate all images in `--data_dir`. 
By default it will use the recent DeepGestalt model with a batch size of 1 on the GPU ("saved_models/s2_gmdb_adam_DeepGestalt_e310_ReLU_BN_bs280.pt").
It will load all images in the `--data_dir`, which by default is set to `../data/GestaltMatcherDB/images_cropped`

When retraining a model with different parameters, make sure to check if the `--num_classes` is the same.
Additionally, you can save the face encodings with `--save_encodings` (default=True) to `encodings.csv`
Using these encodings as input for <<INSERT SCRIPT TZUNG>> will allow you to obtain the results listed in the table.