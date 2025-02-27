from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from ClassifierInterface import MnistClassifierInterface
import numpy as np


class ConvolutionalNeuralNetworkClassifier(MnistClassifierInterface):

    def __init__(self):
        self.model = None

    def train(self, X, y):

        self.model = Sequential()
        # Convolutional and pooling layers addition
        input_layer = Input(shape=(28, 28, 1))
        conv_layer = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        pool_layer = MaxPooling2D(pool_size=(2, 2))
        # Flat layer addition
        flat_layer = Flatten()
        # Dense layer addition
        dense_layer_one = Dense(64, activation='relu')
        dense_layer_two = Dense(10, activation='softmax')
        self.model.add(input_layer)
        self.model.add(conv_layer)
        self.model.add(pool_layer)
        self.model.add(flat_layer)
        self.model.add(dense_layer_one)
        self.model.add(dense_layer_two)
        # Model compilation
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        self.model.fit(X, y)

    def predict(self, X):
        if self.model is None:
            return None
        else:
            probabilities = self.model.predict(X)
            return np.argmax(probabilities, axis=1)
