# GestaltMatcher
Unofficial implementation of GestaltMatcher as described in https://www.medrxiv.org/content/10.1101/2020.12.28.20248193v2.

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

## Results
The table below hold the results from the original paper and the reproductions provided in this repo (Enc-healthy, Enc-GMDB, Enc-GMDB softmax).
| Test set | Model | Top-1 | Top-5 | Top-10 | Top-30 |
|:-|:-|:-|:-|:-|:-|
| GMDB-frequent | Enc-GMDB (softmax) | 29.98% | 48.31% | 66.30% | 81.71% |
| \*GMDB-frequent | Enc-GMDB (softmax) | 25.10% | 46.67% | 61.61% | 78.46% |
| GMDB-frequent | Enc-GMDB | 21.86% | 40.09% | 53.59% | 74.28% |
| \*GMDB-frequent | Enc-GMDB | 20.64% | 41.50% | 54.10% | 71.67% |
| GMDB-frequent | Enc-healthy | 17.04% | 33.26% | 44.03% | 63.46% |
| \*GMDB-frequent | Enc-healthy | 16.29% | 36.00% | 42.52% | 64.31% |
|---|---|---|---|---|---|
| GMDB-rare | Enc-GMDB | 18.51% | 37.50% | 47.36% | 71.93% |
| \*GMDB-rare | Enc-GMDB | 15.78% | 34.68% | 46.19% | 69.24% |
| GMDB-rare | Enc-healthy | 14.85% | 30.53% | 40.43% | 61.65% |
| \*GMDB-rare | Enc-healthy | 12.79% | 29.06% | 40.19% | 61.50% |
|---|---|---|---|---|---|
| GMDB-frequent | Enc-GMDB | 20.98% | 38.25% | 51.05% | 71.37% |
| \*GMDB-frequent | Enc-GMDB | 18.34% | 40.19% | 52.21% | 68.17% |
| GMDB-frequent | Enc-healthy | 15.14% | 31.14% | 42.20% | 62.48% |
| \*GMDB-frequent | Enc-healthy | 15.89% | 35.94% | 41.00% | 62.62% |
|---|---|---|---|---|---|
| GMDB-frequent | Enc-GMDB | 8.47% | 18.33% | 23.19% | 37.62% |
| \*GMDB-frequent | Enc-GMDB | 7.77% | 14.86% | 20.42% | 36.12% |
| GMDB-frequent | Enc-healthy | 7.10% | 14.36% | 19.34% | 30.77% |
| \*GMDB-frequent | Enc-healthy | 6.75% | 13.08% | 17.15% | 28.29% |
