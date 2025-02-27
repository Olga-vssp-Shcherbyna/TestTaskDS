from sklearn.ensemble import RandomForestClassifier
from ClassifierInterface import MnistClassifierInterface


class RandomForestMnistClassifier(MnistClassifierInterface):

    def __init__(self):
        self.model = None

    def train(self, X, y):
        self.model = RandomForestClassifier(n_estimators=150, random_state=42)
        self.model.fit(X, y)

    def predict(self, X):
        if self.model is None:
            return None
        else:
            return self.model.predict(X)
