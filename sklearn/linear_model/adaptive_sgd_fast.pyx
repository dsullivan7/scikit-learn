# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel (partial_fit support)
#         Rob Zinkov (passive-aggressive)
#         Lars Buitinck
#
# Licence: BSD 3 clause


import numpy as np
import sys
from time import time

cimport cython
from libc.math cimport exp, log, sqrt, pow, fabs
cimport numpy as np
from sgd_fast cimport LossFunction, Hinge
cdef extern from "sgd_fast_helpers.h":
    bint skl_isfinite(double) nogil

from sklearn.utils.seq_dataset cimport SequentialDataset

np.import_array()

# Penalty constants
DEF NO_PENALTY = 0
DEF L1 = 1
DEF L2 = 2
DEF ELASTICNET = 3

# Learning rate constants
DEF CONSTANT = 1
DEF OPTIMAL = 2
DEF INVSCALING = 3
DEF PA1 = 4
DEF PA2 = 5


def plain_sgd(np.ndarray[double, ndim=1, mode='c'] weights,
              double intercept,
              LossFunction loss,
              int penalty_type,
              double alpha, double C,
              double l1_ratio,
              SequentialDataset dataset,
              int n_iter, int fit_intercept,
              int verbose, bint shuffle, np.uint32_t seed,
              double weight_pos, double weight_neg,
              int learning_rate, double eta0,
              double power_t,
              double counter=1.0,
              double eps0=0.1,
              double intercept_decay=1.0):
    """Plain SGD for generic loss functions and penalties.

    Parameters
    ----------
    weights : ndarray[double, ndim=1]
        The allocated coef_ vector.
    intercept : double
        The initial intercept.
    loss : LossFunction
        A concrete ``LossFunction`` object.
    penalty_type : int
        The penalty 2 for L2, 1 for L1, and 3 for Elastic-Net.
    alpha : float
        The regularization parameter.
    C : float
        Maximum step size for passive aggressive.
    l1_ratio : float
        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
    dataset : SequentialDataset
        A concrete ``SequentialDataset`` object.
    n_iter : int
        The number of iterations (epochs).
    fit_intercept : int
        Whether or not to fit the intercept (1 or 0).
    verbose : int
        Print verbose output; 0 for quite.
    shuffle : boolean
        Whether to shuffle the training data before each epoch.
    weight_pos : float
        The weight of the positive class.
    weight_neg : float
        The weight of the negative class.
    seed : np.uint32_t
        Seed of the pseudorandom number generator used to shuffle the data.
    learning_rate : int
        The learning rate:
        (1) constant, eta = eta0
        (2) optimal, eta = 1.0/(alpha * t).
        (3) inverse scaling, eta = eta0 / pow(t, power_t)
        (4) Passive Agressive-I, eta = min(alpha, loss/norm(x))
        (5) Passive Agressive-II, eta = 1.0 / (norm(x) + 0.5*alpha)
    eta0 : double
        The initial learning rate.
    eps0 : double
        The regret for the adaptive learning rate
    power_t : double
        The exponent for inverse scaling learning rate.
    counter : double
        Initial state of the learning rate. This value is equal to the
        iteration count except when the learning rate is set to `optimal`.
        Default: 1.0.

    Returns
    -------
    weights : array, shape=[n_features]
        The fitted weight vector.
    intercept : float
        The fitted intercept term.
    """
    standard_weights, standard_intercept = _plain_sgd(weights,
                          intercept,
                          None,
                          0,
                          loss,
                          penalty_type,
                          alpha, C,
                          l1_ratio,
                          dataset,
                          n_iter, fit_intercept,
                          verbose, shuffle, seed,
                          weight_pos, weight_neg,
                          learning_rate, eta0,
                          power_t,
                          counter,
                          intercept_decay,
                          eps0,
                          0)
    return standard_weights, standard_intercept


def average_sgd(np.ndarray[double, ndim=1, mode='c'] weights,
                double intercept,
                np.ndarray[double, ndim=1, mode='c'] average_weights,
                double average_intercept,
                LossFunction loss,
                int penalty_type,
                double alpha, double C,
                double l1_ratio,
                SequentialDataset dataset,
                int n_iter, int fit_intercept,
                int verbose, bint shuffle, np.uint32_t seed,
                double weight_pos, double weight_neg,
                int learning_rate, double eta0,
                double power_t,
                double counter=1.0,
                double intercept_decay=1.0,
                double eps0=0.1,
                int average=1):
    """Average SGD for generic loss functions and penalties.

    Parameters
    ----------
    weights : ndarray[double, ndim=1]
        The allocated coef_ vector.
    intercept : double
        The initial intercept.
    average_weights : ndarray[double, ndim=1]
        The average weights as computed for ASGD
    average_intercept : double
        The average intercept for ASGD
    loss : LossFunction
        A concrete ``LossFunction`` object.
    penalty_type : int
        The penalty 2 for L2, 1 for L1, and 3 for Elastic-Net.
    alpha : float
        The regularization parameter.
    C : float
        Maximum step size for passive aggressive.
    l1_ratio : float
        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
    dataset : SequentialDataset
        A concrete ``SequentialDataset`` object.
    n_iter : int
        The number of iterations (epochs).
    fit_intercept : int
        Whether or not to fit the intercept (1 or 0).
    verbose : int
        Print verbose output; 0 for quite.
    shuffle : boolean
        Whether to shuffle the training data before each epoch.
    weight_pos : float
        The weight of the positive class.
    weight_neg : float
        The weight of the negative class.
    seed : np.uint32_t
        Seed of the pseudorandom number generator used to shuffle the data.
    learning_rate : int
        The learning rate:
        (1) constant, eta = eta0
        (2) optimal, eta = 1.0/(alpha * t).
        (3) inverse scaling, eta = eta0 / pow(t, power_t)
        (4) Passive Agressive-I, eta = min(alpha, loss/norm(x))
        (5) Passive Agressive-II, eta = 1.0 / (norm(x) + 0.5*alpha)
    eta0 : double
        The initial learning rate.
    eps0 : double
        The regret for the adaptive learning rate
    power_t : double
        The exponent for inverse scaling learning rate.
    counter : double
        Initial state of the learning rate. This value is equal to the
        iteration count except when the learning rate is set to `optimal`.
        Default: 1.0.
    average : int
        The number of iterations before averaging starts. average=1 is
        equivalent to averaging for all iterations.

    Returns
    -------
    weights : array, shape=[n_features]
        The fitted weight vector.
    intercept : float
        The fitted intercept term.
    average_weights : array shape=[n_features]
        The averaged weights accross iterations
    average_intercept : float
        The averaged intercept accross iterations
    """
    return _plain_sgd(weights,
                      intercept,
                      average_weights,
                      average_intercept,
                      loss,
                      penalty_type,
                      alpha, C,
                      l1_ratio,
                      dataset,
                      n_iter, fit_intercept,
                      verbose, shuffle, seed,
                      weight_pos, weight_neg,
                      learning_rate, eta0,
                      power_t,
                      counter,
                      intercept_decay,
                      eps0,
                      average)


def _plain_sgd(np.ndarray[double, ndim=1, mode='c'] weights,
               double intercept,
               np.ndarray[double, ndim=1, mode='c'] average_weights,
               double average_intercept,
               LossFunction loss,
               int penalty_type,
               double alpha, double C,
               double l1_ratio,
               SequentialDataset dataset,
               int n_iter, int fit_intercept,
               int verbose, bint shuffle, np.uint32_t seed,
               double weight_pos, double weight_neg,
               int learning_rate, double eta0,
               double power_t,
               double counter=1.0,
               double intercept_decay=1.0,
               double eps0=0.1,
               int average=0):

    # get the data information into easy vars
    cdef Py_ssize_t n_samples = dataset.n_samples
    cdef Py_ssize_t n_features = weights.shape[0]

    cdef double* w_ptr = &weights[0]
    cdef double *x_data_ptr = NULL
    cdef int *x_ind_ptr = NULL
    cdef double* ps_ptr = NULL

    # helper variables
    cdef bint infinity = False
    cdef int xnnz
    cdef double eta = 0.0
    cdef double p = 0.0
    cdef double sumloss = 0.0
    cdef double y = 0.0
    cdef double sample_weight
    cdef double class_weight = 1.0
    cdef unsigned int count = 0
    cdef unsigned int epoch = 0
    cdef unsigned int i = 0
    cdef int is_hinge = isinstance(loss, Hinge)
    cdef double optimal_init = 0.0
    cdef double dloss = 0.0
    cdef double MAX_DLOSS = 1e12

    # the accumulation of the sums of gradients
    cdef np.ndarray[double, ndim = 1, mode = "c"] accu_grad = \
        np.zeros((n_features,), dtype=np.float64, order="c")
    cdef double* accu_grad_ptr = <double*> accu_grad.data

    # the accumulation of the sums of squared gradients
    cdef np.ndarray[double, ndim = 1, mode = "c"] accu_sq_grad = \
        np.zeros((n_features,), dtype=np.float64, order="c")
    cdef double* accu_sq_grad_ptr = <double*> accu_sq_grad.data

    t_start = time()
    with nogil:
        for epoch in range(n_iter):
            if verbose > 0:
                with gil:
                    print("-- Epoch %d" % (epoch + 1))
            if shuffle:
                dataset.shuffle(seed)
            for i in range(n_samples):
                dataset.next(&x_data_ptr,
                             &x_ind_ptr,
                             &xnnz,
                             &y,
                             &sample_weight)

                if counter > 1.0:
                    for j in range(xnnz):
                        idx = x_ind_ptr[j]
                        update_weights(idx, eta0, eps0, alpha, counter, w_ptr,
                                       accu_grad_ptr, accu_sq_grad_ptr)



                if verbose > 0:
                    sumloss += loss.loss(p, y)

                class_weight = weight_pos if y > 0.0 else weight_neg

                p = dot(w_ptr, x_data_ptr, x_ind_ptr, xnnz)
                dloss = loss._dloss(p, y)

                # clip dloss with large values to avoid numerical
                # instabilities
                if dloss < -MAX_DLOSS:
                    dloss = -MAX_DLOSS
                elif dloss > MAX_DLOSS:
                    dloss = MAX_DLOSS

                if dloss != 0.0:
                    for j in range(xnnz):
                        idx = x_ind_ptr[j]
                        val = x_data_ptr[j]
                        grad = val * dloss
                        accu_grad[idx] += grad
                        accu_sq_grad[idx] += grad * grad

                counter += 1
                count += 1

            # report epoch information
            if verbose > 0:
                with gil:
                    print("Norm: %.2f, NNZs: %d, "
                          "Bias: %.6f, T: %d, Avg. loss: %.6f"
                          % (sqnorm(w_ptr, n_features),
                             weights.nonzero()[0].shape[0],
                             intercept, count, sumloss / count))
                    print("Total training time: %.2f seconds."
                          % (time() - t_start))

            # floating-point under-/overflow check.
            if (not skl_isfinite(intercept)
                or any_nonfinite(w_ptr, n_features)):
                infinity = True
                break

    for j in xrange(n_features):
        update_weights(j, eta0, eps0, alpha, counter, w_ptr,
                       accu_grad_ptr, accu_sq_grad_ptr)

    if infinity:
        raise ValueError(("Floating-point under-/overflow occurred at epoch"
                          " #%d. Scaling input data with StandardScaler or"
                          " MinMaxScaler might help.") % (epoch + 1))


    return weights, intercept

cdef void update_weights(int idx, double eta, double eps0, double alpha,
                         double counter, double* w_ptr, double* accu_grad_ptr,
                         double* accu_sq_grad_ptr) nogil:
    eta_t = eta * (counter - 1)
    denom = sqrt(accu_sq_grad_ptr[idx] + eps0) + eta_t * alpha
    w_ptr[idx] = -eta * accu_grad_ptr[idx] / denom

cdef bint any_nonfinite(double *w, int n) nogil:
    for i in range(n):
        if not skl_isfinite(w[i]):
            return True
    return 0

cdef double dot(double* w_ptr,
                double* x_data_ptr,
                int* x_ind_ptr,
                int xnnz) nogil:
    dot_product = 0.0
    for j in range(xnnz):
        idx = x_ind_ptr[j]
        val = x_data_ptr[j]
        weight = w_ptr[idx]
        dot_product += val * weight

    return dot_product


cdef double sqnorm(double * data_ptr, int xnnz) nogil:
    cdef double norm = 0.0
    cdef double z
    for j in range(xnnz):
        z = data_ptr[j]
        norm += z * z
    return norm
