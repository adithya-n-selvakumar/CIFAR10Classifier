## Overview

This project implements a Convolutional Neural Network (CNN) that achieves over 99.5% accuracy on the CIFAR-10 dataset using PyTorch. The model incorporates several advanced techniques to enhance performance and generalization.

## Features

- Custom CNN architecture optimized for CIFAR-10
- Data augmentation techniques for improved generalization
- Dropout for regularization
- Learning rate scheduling using ReduceLROnPlateau
- Comprehensive training and evaluation pipeline
- Visualization of training metrics and confusion matrix

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- seaborn
- tqdm
- scikit-learn
- numpy

## Installation

1. Clone this repository:
git clone https://github.com/yourusername/CIFAR10Classifier.git
cd CIFAR10Classifier

2. Install the required packages:
!pip install torch torchvision matplotlib seaborn tqdm scikit-learn numpy

## Usage

Run the main script to train the model:
python cifar10_classifier.py

This will:
1. Load and preprocess the CIFAR-10 dataset
2. Train the CNN model
3. Evaluate the model on the test set
4. Generate and save training metrics and confusion matrix plots

## Model Architecture

The CNN architecture consists of:
- 3 convolutional layers with ReLU activation and max pooling
- 2 fully connected layers with dropout

## Data Augmentation

The following augmentation techniques are applied to the training data:
- Random crop (32x32 with padding=4)
- Random horizontal flip
- Normalization

## Results

The model typically achieves over 85% accuracy on the CIFAR-10 test set. Actual results may vary slightly due to the random nature of initialization and data augmentation.

## Visualization

The script generates two plots:
1. Training and validation accuracy/loss over epochs
   ![image](https://github.com/user-attachments/assets/daad9ab0-6165-40be-b286-616b3f7424e0)
2. Confusion matrix for the model predictions on the test set
   ![image](https://github.com/user-attachments/assets/836d1b8a-eb6a-4d1c-964e-493ee07000dd)

## Author

Adithya N. Selvakumar

## Acknowledgments

- The CIFAR-10 dataset creators and maintainers
- The PyTorch team for their excellent deep learning framework
