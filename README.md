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

The Lumper and Splitter Analysis is designed to assess the similarity between different cohorts. This method helps determine whether two given cohorts should be merged (lumped) or treated as separate entities (split). The analysis is performed by comparing the similarity between the two cohorts and their relationship with a control distribution.

For example, this approach can be applied to evaluate the similarity between:

* **Various genetic disorders with overlapping or distinct facial patterns**: For example, we proved that patients with NAA10 and NAA15 share similar facial phenotypes (https://www.nature.com/articles/s41431-023-01368-y). On the other hand, we can also demonstrate that one disorder appears distinct from another.

* **Different mutation types or positions within a single gene**: For example, we can show that patients with C-terminal truncation exhibit distinct facial features compared to those with N-terminal truncation.

To reporduce the results in our paper, please contact us to get the corresponding metadata and encoding and store them in "./data" folder.

This section contains various statistical and visualization analyses:

* **Statistics Analysis**: Currently written in R, with plans to migrate to Python in the near future. To run the statistical analysis follow the next steps:
  1. Calculate the distances between each pair of images from the GMDB while excluding images that have been included in the training of GestaltMatcher-Arc:
     ```
     Rscript 0_calculate_distances.R ./data/gmdb_embeddings_wo_pleiotropy_v1.0.3_15012023_remove_genes.p ./data/gmdb_syndromes_v1.0.3_wo_pleiotropy.tsv ./data/image_metadata_v1.0.3.tsv ./data/gmdb_frequent_gallery_images_v1.0.3.csv outcomes/distances
     ```
  3. Conduct roc-analysis via 5-fold cross-validation to derive threshold:
     ```
     Rscript 1_control_distr_roc.R ./data/gmdb_embeddings_wo_pleiotropy_v1.0.3_15012023_remove_genes.p ./data/gmdb_syndromes_v1.0.3_wo_pleiotropy.tsv ./data/image_metadata_v1.0.3.tsv ./data/gmdb_frequent_gallery_images_v1.0.3.csv outcomes/distances.RData outcomes/roc
     ```
  4. Validate derived threshold and subsampling approach on validation set:
     ```
     Rscript scripts/2_accuracy_splitting_lumping.R outcomes/roc.RData outcomes/distances.RData outcomes/lumping_splitting
     ```
  6. Compare two cohorts (lumping and splitting, e.g. MN1 C-terminal vs. N-terminal):
      ```
     Rscript 3_compare_two_cohorts.R ./data/MCTT_embeddings_wo_pleiotropy_v1.0.3_15012023.p ./data/MNTT_embeddings_wo_pleiotropy_v1.0.3_15012023.p MN1 C-terminal N-terminal outcomes/roc.RData outcomes/lumping_splitting.RData outcomes/MCTT_MNTT
      ```
  8. Compare one cohort to random controls (eg. MN1 N-terminal vs. random):
     ```
     Rscript 4_cohort_vs_random.R ./data/gmdb_embeddings_wo_pleiotropy_v1.0.3_15012023_remove_genes.p ./data/gmdb_syndromes_v1.0.3_wo_pleiotropy.tsv ./data/image_metadata_v1.0.3.tsv ./data/gmdb_frequent_gallery_images_v1.0.3.csv outcomes/distances.RData ./data/MNTT_embeddings_wo_pleiotropy_v1.0.3_15012023.p N-terminal outcomes/N_vs_random
     ```
     
* **tSNE Plot**: Provides a 2D visualization of image distributions using Python.
     ```
     python lumping_splitting_tsne.py
     ```
     The results will be shown in analysis_out/MCTT-output-revision.

* **Pairwise Rank**: Compares the rank of a given patient with controls (patients with other disorders) using Python.
     ```
     python lumping_splitting_tsne.py
     ```
     The results will be shown in analysis_out/MCTT-output-revision.

## Contact
Dr. Tzung-Chien Hsieh

Email: thsieh@uni-bonn.de or la60312@gmail.com

## License
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc/4.0/)
