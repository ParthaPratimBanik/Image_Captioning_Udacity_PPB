# Image_Captioning_Udacity_PPB
This project is on <b>`Generating Caption from Image`</b>, launched by Udacity, Computer Vision Nanodegree program.

## Dataset
In this project, Microsoft **C**ommon **O**bjects in **CO**ntext (MS COCO) dataset is used. It is a large-scale dataset for scene understanding. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.
![Sample Dog Output](images/coco-examples.jpg)
You can read more about the dataset on the [website](http://cocodataset.org/#home) or in the [research paper](https://arxiv.org/pdf/1405.0312.pdf).

To explore the dataset and take the preparation of the project, please see `0_Dataset.ipynb` file.

## Exploring MS COCO Dataset
To explore the MS COCO dataset, please see `1_Preliminaries.ipynb` file.

## CNN-RNN model
To know the CNN-RNN model architechture, please see `model.py` file. 

## Training CNN-RNN Model
To know the training parameters, and optimizer, please see `2_Training.ipynb` file.

## Optimizer and Loss Function Selection
To infer the trained CNN-RNN model on test dataset, please see `3_inference.ipynb` file.