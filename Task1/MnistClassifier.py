from sklearn.metrics import accuracy_score

from RandomForest import RandomForestMnistClassifier
from FeedForwardNeuralNetwork import FeedForwardNeuralNetworkClassifier
from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetworkClassifier
from data_loaders import train_images, train_labels, test_images, test_labels, train_images_cnn, test_images_cnn


class MnistClassifier:

    def __init__(self, algorithm):
        if algorithm == 'rf':
            self.model = RandomForestMnistClassifier()
        elif algorithm == 'nn':
            self.model = FeedForwardNeuralNetworkClassifier()
        elif algorithm == 'cnn':
            self.model = ConvolutionalNeuralNetworkClassifier()
        else:
            raise ValueError("Algorithm is not found")

        self.train_labels = train_labels
        self.test_labels = test_labels
        if algorithm == 'rf' or algorithm == 'nn':
            self.train_images = train_images
            self.test_images = test_images
        elif algorithm == 'cnn':
            self.train_images = train_images_cnn
            self.test_images = test_images_cnn

    def train(self):
        self.model.train(self.train_images, self.train_labels)

    def predict(self):
        return self.model.predict(self.test_images)

    def accuracy(self):
        return accuracy_score(self.test_labels, self.predict())



