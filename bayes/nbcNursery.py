import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class NBCDiscreteForNursery(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1):
        self.alpha_ = alpha

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y

        # Унікальні класи
        self.labels_ = labels = np.unique(y)

        # Ініціалізація структур для зберігання частот
        classFeatureCounts = {}
        featureCounts = [{} for _ in range(X.shape[1])]
        self.label_counts_ = {}

        # Загальна кількість елементів
        self.total_count_ = len(y)

        for label in labels:
            classFeatureCounts[label] = [{} for _ in range(X.shape[1])]
            self.label_counts_[label] = 0

        # Підрахунок частот
        for xi, yi in zip(X, y):
            self.label_counts_[yi] += 1
            for index, value in enumerate(xi):
                feature_class_count = classFeatureCounts[yi][index]
                feature_class_count[value] = feature_class_count.get(value, 0) + 1
                feature_counts = featureCounts[index]
                feature_counts[value] = feature_counts.get(value, 0) + 1

        self.classFeatureCounts_ = classFeatureCounts
        self.featureCounts_ = featureCounts
        return self

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.labels_[np.argmax(probs, axis=1)]

    def predict_proba(self, X):
        result = []
        for xi in X:
            class_probs = []
            for label in self.labels_:
                prob = self.label_counts_[label] / self.total_count_
                for index, value in enumerate(xi):
                    feature_class_count = self.classFeatureCounts_[label][index].get(value, 0)
                    unique_feature_values = len(self.featureCounts_[index])
                    prob *= (feature_class_count + self.alpha_) / (
                        self.label_counts_[label] + unique_feature_values * self.alpha_
                    )
                class_probs.append(prob)
            total_prob = sum(class_probs)
            class_probs = [p / total_prob for p in class_probs]
            result.append(class_probs)
        return np.array(result)