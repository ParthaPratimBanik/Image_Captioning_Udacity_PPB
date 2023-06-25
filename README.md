# Image_Captioning_Udacity_PPB
---
This project is on <b>`Generating Caption from Image`</b>, launched by Udacity, Computer Vision Nanodegree program.

## Dataset
---
In this project, Microsoft **C**ommon **O**bjects in **CO**ntext (MS COCO) dataset is used. It is a large-scale dataset for scene understanding. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.
![Sample Dog Output](images/coco-examples.jpg)
You can read more about the dataset on the [website](http://cocodataset.org/#home) or in the [research paper](https://arxiv.org/pdf/1405.0312.pdf).

To explore the dataset and take the preparation of the project, please see `0_Dataset.ipynb` file.


## Exploring MS COCO Dataset
---
To use the dataset, please follow the instruction of [cocoapi](https://github.com/cocodataset/cocoapi/tree/master). According to the instruction, please download the full dataset (images + annotations) and maintain the dataset directory structure like following:

`Directory Tree of Dataset` 
```
├───opt
│   └───cocoapi
│       ├───annotations
│       └───images
│           ├───test2014
│           ├───train2014
│           └───val2014
```

After installing the `API`, please run "make" under coco/PythonAPI. For details, please see `1_Preliminaries.ipynb` file.

Now,
- By using `pip install nltk`, install `nltk` python package in the environment.

## CNN-RNN model
To know the CNN-RNN model architechture, please see `model.py` file. 

## Training CNN-RNN Model
To know the training parameters, and optimizer, please see `2_Training.ipynb` file.

## Optimizer and Loss Function Selection
To infer the trained CNN-RNN model on test dataset, please see `3_inference.ipynb` file.