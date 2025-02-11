# GestaltMatcher
GestaltMatcher is an AI-driven tool for deep facial phenotyping to aid in the diagnosis of ultra-rare genetic disorders. This repository serves as the main landing page for all tools related to GestaltMatcher, developed by the Institute for Genomic Statistics and Bioinformatics (IGSB) at the University of Bonn.


## Publications

GestaltMatcher was first introduced in Nature Genetics:

* Hsieh, T.-C. et al. GestaltMatcher facilitates rare disease matching using facial phenotype descriptors. Nat. Genet. 54, 349–357 (2022). (Nature Genetics)

Later, its performance was further enhanced in WACV 2023:

* Hustinx, A. et al. Improving deep facial phenotyping for ultra-rare disorder verification using model ensembles. in 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (IEEE, 2023). doi:10.1109/wacv56688.2023.00499. (WACV 2023)

## Submodules

GestaltMatcher consists of three core submodules:

* GestaltEngine-FaceCropper (Face Cropper) – Extracts and aligns facial regions from images. (submodule)


* GestaltMatcher-Arc (Model Training and Encoding) – Utilizes deep learning to analyze and encode facial features associated with genetic syndromes. (submodule)

* Evaluation (This Repository) – Assesses the model's performance on benchmark datasets.

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

## Lumper and Splitter Analysis

This section contains various statistical and visualization analyses:

* Statistics Analysis: Currently written in R, with plans to migrate to Python in the near future.

* tSNE Plot: Provides a 2D visualization of image distributions using Python.

* Pairwise Rank: Compares the rank of a given patient with controls (patients with other disorders) using Python.

## Contact
Dr. Tzung-Chien Hsieh

Email: thsieh@uni-bonn.de or la60312@gmail.com

## License
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc/4.0/)
