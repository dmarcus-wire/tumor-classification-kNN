# Tumor Classification
>Goal: use K-Nearest Neighbors to classify images of tumors as: benign, malignant, or normal.

1. Starting with the kaggle dataset https://www.kaggle.com/alifrahman/modiified, use k-NN to classify imagery of tumors as benign, malignent or normal.
1. Dataset was collapsed from train and test with separate subclasses into 1 dataset tumors with separate subclasses.
    - benign = 351 images sized @ 203x170 px
    - malignant = 351 images sized @ 203x170 px
    - normal = 351 images sized @ 203x170 px
1. Data splitting will be handled by scikit-learn module.
1. Project structure was created.
1. Datasets (images) were copied into the respective classes of tumors.
1. Project structure is as follows: 

```
tree .
.
├── README.md
├── dataset
│         ├── submodules
│         │         ├── datasets
│         │         │         ├── __init__.py
│         │         │         └── simpledatasetloader.py
│         │         └── preprocessing
│         │             ├── __init__.py
│         │             └── simplepreprocessor.py
│         └── tumors
│             ├── benign
│             ├── malignant
│             └── normal
└── requirements.txt
```

This is just an example of using KNN, but it won't scale due to storing training data beyond thousands of images. See Parameterized learning.