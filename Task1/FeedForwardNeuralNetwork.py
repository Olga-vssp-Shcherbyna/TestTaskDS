from sklearn.neural_network import MLPClassifier
from ClassifierInterface import MnistClassifierInterface


class FeedForwardNeuralNetworkClassifier(MnistClassifierInterface):

    def __init__(self):
        self.model = None

    def train(self, X, y):
        self.model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=200)
        self.model.fit(X, y)

    def predict(self, X):
        if self.model is None:
            return None
        else:
            return self.model.predict(X)
