from data_loader import load_normalized_data


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = list(X_train)
        self.y_train = list(y_train)

    def _euclidean_distance(self, point_one, point_two):
        squared_differences = [
            (coord_one - coord_two) ** 2
            for coord_one, coord_two in zip(point_one, point_two)
        ]
        return sum(squared_differences) ** 0.5

    def predict(self, X_test):
        predictions = []
        for query_point in X_test:
            distances = [
                self._euclidean_distance(query_point, training_point)
                for training_point in self.X_train
            ]
            indexed_distances = list(enumerate(distances))
            indexed_distances.sort(key=lambda indexed_pair: indexed_pair[1])
            nearest_neighbors = indexed_distances[: self.k]
            nearest_labels = [
                self.y_train[neighbor_index]
                for neighbor_index, neighbor_distance in nearest_neighbors
            ]
            majority_label = max(set(nearest_labels), key=nearest_labels.count)
            predictions.append(majority_label)
        return predictions

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        y_test = list(y_test)
        correct_count = sum(
            1
            for predicted_label, actual_label in zip(predictions, y_test)
            if predicted_label == actual_label
        )
        return correct_count / len(predictions)

    def grid_search(self, X_train, y_train, X_test, y_test, k_values):
        scores_by_k = {}
        for candidate_k in k_values:
            self.k = candidate_k
            self.fit(X_train, y_train)
            scores_by_k[candidate_k] = self.evaluate(X_test, y_test)
        best_k = max(scores_by_k, key=scores_by_k.get)
        self.k = best_k
        self.fit(X_train, y_train)
        return best_k, scores_by_k[best_k], scores_by_k


if __name__ == "__main__":
    X_normalized, Y, standard_scaler_object = load_normalized_data(
        file_path="bienetre.csv.csv"
    )

    split_index = int(0.8 * len(X_normalized))
    X_train = X_normalized[:split_index]
    X_test = X_normalized[split_index:]
    y_train = list(Y)[:split_index]
    y_test = list(Y)[split_index:]

    knn_model = KNN(k=3)
    knn_model.fit(X_train, y_train)
    accuracy = knn_model.evaluate(X_test, y_test)
    print("Accuracy avec k=3 :", accuracy)

    best_k, best_score, all_scores = knn_model.grid_search(
        X_train, y_train, X_test, y_test, k_values=[1, 3, 5, 7, 9]
    )
    print("Meilleur k :", best_k, "→ score :", best_score)
    print("Tous les scores :", all_scores)
