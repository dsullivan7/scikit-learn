import numpy as np
import math
from sklearn.linear_model import AdaptiveSGDRegressor


def log_dloss(p, y):
    z = p * y
    # approximately equal and saves the computation of the log
    if z > 18.0:
        return math.exp(-z) * -y
    if z < -18.0:
        return -y
    return -y / (math.exp(z) + 1.0)


def sparse_sgd(X, y, eta, n_iter=5):
    weights = np.zeros(X.shape[1])
    accu_gradient = np.zeros(X.shape[1])

    for j in range(n_iter):
        for i, entry in enumerate(X):
            p = np.dot(entry, weights)
            scalar_gradient = log_dloss(p, y[i])
            gradient_vector = scalar_gradient * entry

            accu_gradient += gradient_vector * gradient_vector
            weights -= (eta / np.sqrt(accu_gradient) *
                        gradient_vector)

    return weights


def test_computed_correctly():
    X = np.random.random((5, 4))
    y = np.random.random(5)
    clf = AdaptiveSGDRegressor(eta0=1.0, alpha=1.0, n_iter=1, loss="log")
    clf.fit(X, y)
    w = sparse_sgd(X, y, 1.0, n_iter=1)

    print(clf.coef_)
    print(w)
