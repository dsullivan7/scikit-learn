import numpy as np
import math
from sklearn.linear_model import AdaptiveSGDRegressor
from sklearn.utils.testing import assert_array_almost_equal



def log_dloss(p, y):
    z = p * y
    # approximately equal and saves the computation of the log
    if z > 18.0:
        return math.exp(-z) * -y
    if z < -18.0:
        return -y
    return -y / (math.exp(z) + 1.0)


def sparse_sgd(X, y, eta, alpha, eps0=.1, n_iter=5, dloss=log_dloss):
    n_shape, n_features = X.shape
    weights = np.zeros(X.shape[1])
    accu_gradient = np.zeros(X.shape[1])
    accu_sq_gradient = np.zeros(X.shape[1])
    counter = 1.0
    for j in range(n_iter):
        for i, entry in enumerate(X):
            if counter > 1:
                eta_t = eta * (counter - 1)
                for idx in range(n_features):
                    denom = math.sqrt(accu_sq_gradient[idx] + eps0) + \
                        eta_t * alpha
                    weights[idx] = -eta * accu_gradient[idx] / denom

            p = np.dot(entry, weights)
            scalar_gradient = dloss(p, y[i])

            gradient_vector = scalar_gradient * entry

            accu_gradient += gradient_vector
            accu_sq_gradient += gradient_vector * gradient_vector

            counter += 1

    eta_t = eta * (counter - 1)
    for idx in range(n_features):
        denom = math.sqrt(accu_sq_gradient[idx] + eps0) + eta_t * alpha
        weights[idx] = -eta * accu_gradient[idx] / denom
    return weights


def test_computed_correctly():
    n_samples, n_features = 100, 20
    eta = .01
    alpha = .001
    n_iter = 10
    np.random.seed(44)
    X = np.random.random((n_samples, n_features))
    y = np.random.random(n_samples)

    clf = AdaptiveSGDRegressor(eta0=eta, alpha=alpha, eps0=0.1,
                               n_iter=n_iter, loss="log")
    clf.fit(X, y)
    w = sparse_sgd(X, y, eta, alpha, n_iter=n_iter, eps0=0.1, dloss=log_dloss)

    assert_array_almost_equal(w, clf.coef_, decimal=16)
