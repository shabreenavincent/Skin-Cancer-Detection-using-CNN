# Skin Cancer Detection System Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Model Architecture](#model-architecture)
4. [Data](#data)
5. [How to Run the App](#how-to-run-the-app)
   - [Method 1](#method-1)
   - [Method 2](#method-2)
6. [Libraries Used](#libraries-used)
7. [Files Overview](#files-overview)
   - [skin_cancer_detection.ipynb](#skin_cancer_detectionipynb)
   - [app.py](#apppy)
   - [skin_cancer_detection.py](#skin_cancer_detectionpy)
   - [best_model.h5](#best_modelh5)
8. [CNN Model Summary](#cnn-model-summary)
9. [Future Work](#future-work)
10. [Conclusion](#conclusion)
11. [References](#references)

## Introduction

The primary goal of this project is to develop a Skin Cancer Detection System using Machine Learning Algorithms. The focus is on building an accurate and efficient model to classify skin cancer images based on dermatoscopic features.

## Project Overview

After experimenting with various architectures for the CNN model, it was found that adding the BatchNormalization layer after each Dense and MaxPooling2D layer significantly improved the validation accuracy. Future development plans include creating a mobile application to enhance accessibility.

You can play with Skin Cancer Images [here](https://skin-cancer-detection-cnn.herokuapp.com/).

## Model Architecture

![Model Architecture](https://github.com/charanhu/Skin-Cancer-Detection-MNIST/blob/main/model_architecture.png)

## Data

The dataset used for this project is available on [Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000). It consists of dermatoscopic images for classifying skin cancer into seven different classes.

## How to Run the App

### Method 1

1. Run the `app.py` file.
2. Open your browser and navigate to `http://localhost:5000/`.
3. Use the Upload button to browse and upload the image you want to classify.
4. Click the Submit button to get the results.

### Method 2

1. Deploy the application to Azure Web App or Heroku through the GitHub repository.
2. Open the URL generated after deployment in your browser.
3. Use the Upload button to browse and upload the image you want to classify.
4. Click the Submit button to get the results.

## Libraries Used

- `numpy`
- `keras`
- `tensorflow-cpu==2.5.0`
- `pandas`
- `matplotlib`
- `pillow`
- `flask`
- `seaborn`
- `gunicorn`

## Files Overview

### skin_cancer_detection.ipynb

This Jupyter notebook is used to define and train the model.

### app.py

This is the Flask app that needs to be running to use the web application.

### skin_cancer_detection.py

This file contains the definition of the CNN model.

### best_model.h5

This file contains the weights of the best-performing model.

## CNN Model Summary

```python
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 16)        448       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 16)        0         
_________________________________________________________________
batch_normalization (BatchNo (None, 14, 14, 16)        64        
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 12, 12, 32)        4640      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 10, 10, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 5, 5, 64)          256       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 3, 3, 128)         73856     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 1, 1, 256)         295168    
_________________________________________________________________
flatten (Flatten)            (None, 256)               0         
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               65792     
_________________________________________________________________
batch_normalization_2 (Batch (None, 256)               1024      
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               32896     
_________________________________________________________________
batch_normalization_3 (Batch (None, 128)               512       
_________________________________________________________________
dense_2 (Dense)              (None, 64)                8256      
_________________________________________________________________
batch_normalization_4 (Batch (None, 64)                256       
_________________________________________________________________
dropout_2 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 32)                2080      
_________________________________________________________________
batch_normalization_5 (Batch (None, 32)                128       
_________________________________________________________________
dense_4 (Dense)              (None, 7)                 231       
=================================================================
Total params: 504,103
Trainable params: 502,983
Non-trainable params: 1,120
_________________________________________________________________



Conclusion
This Skin Cancer Detection System demonstrates the potential of machine learning algorithms, particularly convolutional neural networks, in medical image classification. The project not only provides an accurate classification of skin cancer images but also sets the foundation for future developments, including mobile applications.

References
K. Mader, "Skin Cancer MNIST: HAM10000," Kaggle. Available: Kaggle Link.
Additional references related to CNNs and medical image processing can be listed here.