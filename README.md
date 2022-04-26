# Pediatric_GMSeg

Transfer learning to Improve Pediatric Spinal Cord Gray Matter (GM) segmentation. Final project for CS 4267 (Deep Learning, Spring 2022). 

## Summary:

Current spinal cord GM segmentation methods are trained using adult spinal cord datasets and sub-optimally translate to the pediatric population. This project aims to improves pediatric GM segmentation by conducting transfer learning on a pretrained deep learning model (*sct_deepseg_gm :* [https://arxiv.org/abs/1710.01269](https://arxiv.org/abs/1710.01269)) using a clinical pediatric spinal cord dataset. After model evaluation, it was shown that transfer learning does improve pediatric gray matter segmentation.

## Training Overview

The following models and data combinations were trained: (1) Pretrained model without data augmentation, (2) Model (no pretraining) without data augmentation, (3)  Pretrained model with data augmentation, and (4) Model (no pretraining) with data augmentation. The models were trained on the Vanderbilt ACCRE high-performance cluster and the scripts can be found in the accre_scripts directory. The python scripts were designed to take advantage of GPU computing and parallelization for efficiency. ACCRE requires the following python dependency versions to run: Python (v3.6.3), Tensorflow (v1.8.0), Keras (v2.20), CUDA (v11.10).

## Repository Structure Explained

### Base Directory

- [*Controls.txt*](Controls.txt) and *Controls_second.txt*: Subject IDs for the spinal cord scans
- *Subject_ID_Level_Split.xlsx*: Excel document containing what subject IDs were used in respective training, validation, and testing sets. Used to handle skew in the data.
- *GM_Pre_Processing.ipynb*: Notebook used to pre-process spinal cord scans and conduct GM segmentation
- D*ata_Creation.ipynb*: Notebook used to create the testing, validation, and testing sets

### GM_Models

- *model.py*: Deepseg_gm model using deep dilated CNN as implemented in [https://arxiv.org/abs/1710.01269](https://arxiv.org/abs/1710.01269)
- *deepseg_gm.py*: Interface API for deepseg_gm model
- *challenge_model.hdf5*: sct_deepseg_gm challenge model weights