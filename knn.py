class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = list(y)

    def _euclidean_distance(self, p1, p2):
        return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5

    def predict(self, X):
        predictions = []
        for point in X:
            distances = [self._euclidean_distance(point, x) for x in self.X_train]
            k_idx = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
            k_labels = [self.y_train[i] for i in k_idx]
            predictions.append(max(set(k_labels), key=k_labels.count))
        return predictions

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        y_test = list(y_test)
        correct = sum(1 for i in range(len(predictions)) if predictions[i] == y_test[i])
        return correct / len(predictions)
