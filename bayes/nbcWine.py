import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

data = np.genfromtxt('../wine.data', delimiter=',')

X = data[:, 1:]
y = data[:, 0].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def discretize_data(X, bins, feature_min=None, feature_max=None):
    feature_min = np.min(X, axis=0) if feature_min is None else feature_min
    feature_max = np.max(X, axis=0) if feature_max is None else feature_max
    bin_edges = [
        np.linspace(feature_min[i], feature_max[i], bins + 1) for i in range(X.shape[1])
    ]
    X_discrete = np.zeros_like(X, dtype=int)
    for i, edges in enumerate(bin_edges):
        X_discrete[:, i] = np.digitize(X[:, i], edges[:-1], right=False)
    return X_discrete


bins = 5
X_train_discrete = discretize_data(X_train, bins)
X_test_discrete = discretize_data(X_test, bins, feature_min=np.min(X_train, axis=0), feature_max=np.max(X_train, axis=0))

class NBCDiscreteForWine(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.feature_probs_ = {}
        self.class_probs_ = {}

        for c in self.classes_:
            X_c = X[y == c]
            self.class_probs_[c] = (len(X_c) + self.alpha) / (len(X) + len(self.classes_) * self.alpha)
            self.feature_probs_[c] = [
                np.bincount(X_c[:, i], minlength=np.max(X[:, i]) + 1) + self.alpha
                for i in range(X.shape[1])
            ]

        return self

    def predict_proba(self, X):
        probs = []
        for x in X:
            class_probs = []
            for c in self.classes_:
                prob = np.log(self.class_probs_[c])
                for i, xi in enumerate(x):
                    prob += np.log(self.feature_probs_[c][i][xi] / np.sum(self.feature_probs_[c][i]))
                class_probs.append(prob)
            probs.append(class_probs)
        return np.exp(probs) / np.sum(np.exp(probs), axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


model = NBCDiscreteForWine(alpha=1)
model.fit(X_train_discrete, y_train)

y_train_pred = model.predict(X_train_discrete)
y_test_pred = model.predict(X_test_discrete)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

from sklearn.naive_bayes import MultinomialNB

gaussian_nb_model = MultinomialNB()
gaussian_nb_model.fit(X_train, y_train)

y_train_pred_gnb = gaussian_nb_model.predict(X_train)
y_test_pred_gnb = gaussian_nb_model.predict(X_test)

train_accuracy_gnb = accuracy_score(y_train, y_train_pred_gnb)
test_accuracy_gnb = accuracy_score(y_test, y_test_pred_gnb)

print(f"Discrete NBC Train Accuracy: {train_accuracy:.16f}")
print(f"Discrete NBC Test Accuracy: {test_accuracy:.16f}")
print(f"MultinomialNB Train Accuracy: {train_accuracy_gnb:.16f}")
print(f"MultinomialNB Test Accuracy: {test_accuracy_gnb:.16f}")

