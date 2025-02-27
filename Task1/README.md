# MNIST Classifier

## Introduction

The MNIST dataset is a collection of handwritten digits widely used for training various image processing systems. This project implements three different classifiers to recognize these digits and compare their performance:

- Convolutional Neural Network (CNN)
- Feedforward Neural Network (NN)
- Random Forest (RF)

## Table of Contents
- [Setup Instructions](#setup-instructions)
- [Dataset](#dataset)
- [Running the Project](#running-the-project)
- [Project Structure](#project-structure)
- [Notes](#notes)
- [License](#license)

## Setup Instructions

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- TensorFlow/Keras
- Scikit-learn
- NumPy

You can install dependencies using:

```sh
pip install -r requirements.txt
```

### Dataset

The MNIST dataset files is provided in a ZIP archive 'MNISTdataset.zip'.
Extract the following files into the project directory:

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

### Running the Project

To train and evaluate a model, run:

```sh
python main.py
```

You will be prompted to select an algorithm (`rf`, `nn`, or `cnn`). After running, you can expect the model to output its evaluation metrics.

### Project Structure

- `main.py` - Entry point for running the classification
- `MnistClassifier.py` - Handles model selection and evaluation
- `data_loaders.py` - Loads the MNIST dataset
- `RandomForest.py` - Implements Random Forest classifier
- `FeedForwardNeuralNetwork.py` - Implements a feedforward neural network
- `ConvolutionalNeuralNetwork.py` - Implements a CNN classifier
- `ClassifierInterface.py` - Abstract base class for classifiers

### Notes

- The dataset is in IDX format, which is loaded using `data_loaders.py`.
- The CNN model uses Keras and TensorFlow.
- The NN model is implemented using `MLPClassifier` from scikit-learn.
- The RF model is implemented using `RandomForestClassifier` from scikit-learn.

### License

This project is open-source under the MIT License.
