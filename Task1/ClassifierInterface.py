from abc import ABC, abstractmethod


class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X, y):
        # scratch method for model training
        pass

    @abstractmethod
    def predict(self, X):
        # scratch method for prediction
        pass
