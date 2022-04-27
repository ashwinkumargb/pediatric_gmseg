# Pediatric_GMSeg

**# Pediatric_GMSeg**

Transfer learning to Improve Pediatric Spinal Cord Gray Matter (GM) segmentation. Final project for CS 5267 (Deep Learning, Spring 2022).

**## Summary:**

Current spinal cord GM segmentation methods are trained using adult spinal cord datasets and sub-optimally translate to the pediatric population. This project aims to improves pediatric GM segmentation by conducting transfer learning on a pretrained deep learning model (**sct_deepseg_gm :** [https://arxiv.org/abs/1710.01269](https://arxiv.org/abs/1710.01269)) using a clinical pediatric spinal cord dataset. After model evaluation, it was shown that transfer learning does improve pediatric gray matter segmentation.

The report that details the experiment in more detail can be found [here](Kumar_Transfer_Learning.pdf).

[](model_eval/figures/result_summary.PNG)

**## Training Overview**

The following models and data combinations were trained: (1) Pretrained model without data augmentation, (2) Model (no pretraining) without data augmentation, (3)  Pretrained model with data augmentation, and (4) Model (no pretraining) with data augmentation. The models were trained on the Vanderbilt ACCRE high-performance cluster and the scripts can be found in the accre_scripts directory. The python scripts were designed to take advantage of GPU computing and parallelization for efficiency. ACCRE requires the following python dependency versions to run: Python (v3.6.3), Tensorflow (v1.8.0), Keras (v2.20), CUDA (v11.10).

The datasets used for the models are publicly available [here](https://drive.google.com/drive/folders/1NcCTYn17AmMOj85jrcVG2JA55lBqBiNr?usp=sharing).

**## Repository Structure Explained**

**### Base Directory**

- [*Controls.txt*](Controls.txt) and [*Controls_second.txt*](Controls_second.txt): Subject IDs for the spinal cord scans
- [*Subject_ID_Level_Split.xlsx*](Subject_ID_Level_Split.xlsx): Excel document containing what subject IDs were used in respective training, validation, and testing sets. Used to handle skew in the data.
- [*GM_Pre_Processing.ipynb*](GM_Pre_Processing.ipynb): Notebook used to pre-process spinal cord scans and conduct GM segmentation
- [D*ata_Creation.ipynb*](data_creation.ipynb): Notebook used to create the testing, validation, and testing sets
- [D*ata_Augment.ipynb*](data_augment.ipynb): Notebook used to augment the training and validation data

**### GM_Models Directory**

- [*model.py*](gm_models/model.py): Deepseg_gm model using deep dilated CNN as implemented in [https://arxiv.org/abs/1710.01269](https://arxiv.org/abs/1710.01269)
- [*deepseg_gm.py*](gm_models/deepseg_gm.py): Interface API for deepseg_gm model
- [*challenge_model.hdf5*](gm_models/challenge_model.hdf5): sct_deepseg_gm challenge model weights
- [*gm_model.ipynb*](gm_models/gm_model.ipynb): GM model used for debugging purposes

**### Accre_Scripts Directory**

- [*gm_aug_model_identify_epochs.py*](accre_scripts/gm_aug_model_identify_epochs.py): GM model to understand number of epochs to train for augmented model
- [*gm_aug_model_no_pretrain.py*](accre_scripts/gm_aug_model_no_pretrain.py): GM model on augmented data based on early stopping with no pretraining
- [*gm_aug_model_pretrain.py*](accre_scripts/gm_aug_model_pretrain.py): GM model on augmented data based on early stopping with pretraining
- [*gm_model_no_pretrain.py*](accre_scripts/gm_model_no_pretrain.py): GM model on non-augmented data based on early stopping with no pretraining
- [*gm_model.py*](accre_scripts/gm_model.py): GM model on non-augmented data based on early stopping with pretraining
- Slurm_Scripts: Scripts used to run python files on Vanderbilt ACCRE

**### Model_Eval Directory**

- [*model_eval_combined.ipynb*](model_eval/model_eval_combined.ipynb): Evaluate the different models and plot the testing results from them accordingly. Models were evaluated using various performance metrics.

Transfer learning to Improve Pediatric Spinal Cord Gray Matter (GM) segmentation. Final project for CS 5267 (Deep Learning, Spring 2022). 

## Summary:

Current spinal cord GM segmentation methods are trained using adult spinal cord datasets and sub-optimally translate to the pediatric population. This project aims to improves pediatric GM segmentation by conducting transfer learning on a pretrained deep learning model (*sct_deepseg_gm :* [https://arxiv.org/abs/1710.01269](https://arxiv.org/abs/1710.01269)) using a clinical pediatric spinal cord dataset. After model evaluation, it was shown that transfer learning does improve pediatric gray matter segmentation.

The report that details the experiment in more detail can be found [here](Kumar_Transfer_Learning.pdf).

## Training Overview

The following models and data combinations were trained: (1) Pretrained model without data augmentation, (2) Model (no pretraining) without data augmentation, (3)  Pretrained model with data augmentation, and (4) Model (no pretraining) with data augmentation. The models were trained on the Vanderbilt ACCRE high-performance cluster and the scripts can be found in the accre_scripts directory. The python scripts were designed to take advantage of GPU computing and parallelization for efficiency. ACCRE requires the following python dependency versions to run: Python (v3.6.3), Tensorflow (v1.8.0), Keras (v2.20), CUDA (v11.10).

The datasets used for the models are publicly available [here][https://drive.google.com/drive/folders/1NcCTYn17AmMOj85jrcVG2JA55lBqBiNr?usp=sharing](https://drive.google.com/drive/folders/1NcCTYn17AmMOj85jrcVG2JA55lBqBiNr?usp=sharing).

## Repository Structure Explained

### Base Directory

- [*Controls.txt*](Controls.txt) and [*Controls_second.txt*](Controls_second.txt): Subject IDs for the spinal cord scans
- [*Subject_ID_Level_Split.xlsx*](Subject_ID_Level_Split.xlsx): Excel document containing what subject IDs were used in respective training, validation, and testing sets. Used to handle skew in the data.
- [*GM_Pre_Processing.ipynb*](GM_Pre_Processing.ipynb): Notebook used to pre-process spinal cord scans and conduct GM segmentation
- [D*ata_Creation.ipynb*](data_creation.ipynb): Notebook used to create the testing, validation, and testing sets
- [D*ata_Augment.ipynb*](data_augment.ipynb): Notebook used to augment the training and validation data

### GM_Models Directory

- [*model.py*](gm_models/model.py): Deepseg_gm model using deep dilated CNN as implemented in [https://arxiv.org/abs/1710.01269](https://arxiv.org/abs/1710.01269)
- [*deepseg_gm.py*](gm_models/deepseg_gm.py): Interface API for deepseg_gm model
- [*challenge_model.hdf5*](gm_models/challenge_model.hdf5): sct_deepseg_gm challenge model weights
- [*gm_model.ipynb*](gm_models/gm_model.ipynb): GM model used for debugging purposes

### Accre_Scripts Directory

- [*gm_aug_model_identify_epochs.py*](accre_scripts/gm_aug_model_identify_epochs.py): GM model to understand number of epochs to train for augmented model
- [*gm_aug_model_no_pretrain.py*](accre_scripts/gm_aug_model_no_pretrain.py): GM model on augmented data based on early stopping with no pretraining
- [*gm_aug_model_pretrain.py*](accre_scripts/gm_aug_model_pretrain.py): GM model on augmented data based on early stopping with pretraining
- [*gm_model_no_pretrain.py*](accre_scripts/gm_model_no_pretrain.py): GM model on non-augmented data based on early stopping with no pretraining
- [*gm_model.py*](accre_scripts/gm_model.py): GM model on non-augmented data based on early stopping with pretraining
- Slurm_Scripts: Scripts used to run python files on Vanderbilt ACCRE

### Model_Eval Directory

- [*model_eval_combined.ipynb*](model_eval/model_eval_combined.ipynb): Evaluate the different models and plot the testing results from them accordingly. Models were evaluated using various performance metrics.