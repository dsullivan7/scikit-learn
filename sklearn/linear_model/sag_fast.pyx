# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

from sklearn.utils.seq_dataset cimport SequentialDataset
from .sgd_fast cimport LossFunction

def fast_fit(SequentialDataset dataset,
             np.ndarray[double, ndim=1, mode='c'] weights,
             int n_samples,
             int n_features,
             int n_iter,
             LossFunction loss,
             double eta,
             double alpha
             ):

    cdef double* weights_ptr = &weights[0]
    cdef double *x_data_ptr
    cdef int *x_ind_ptr
    cdef double y
    cdef double sample_weight
    cdef int xnnz
    cdef int idx
    cdef int current_index
    cdef int class_weight
    cdef int counter
    cdef double full_gradient
    cdef bint* seen = <bint*> malloc(n_samples * sizeof(bint))
    cdef double num_seen = 0.0
    cdef double* sum_gradient = <double*> malloc(n_features *
                                                 sizeof(double))
    cdef double** gradient_memory = <double**> malloc(n_samples *
                                                      sizeof(double))
    for i in range(n_samples):
        gradient_memory[i] = <double*> malloc(n_features * sizeof(double))
        for j in range(n_features):
            gradient_memory[i][j] = 0.0

    for i in range(n_samples):
        seen[i] = False

    for i in range(n_features):
        sum_gradient[i] = 0.0

    cdef double intercept = 0.0

    with nogil:
        for i in range(n_iter * n_samples):
            # extract a random sample
            current_index = dataset.random(&x_data_ptr,
                                           &x_ind_ptr,
                                           &xnnz,
                                           &y,
                                           &sample_weight)

            # update the number of samples seen and the seen array
            if not seen[current_index]:
                num_seen += 1.0
                seen[current_index] = True

            if y > 0.0:
                class_weight = 1
            else:
                class_weight = -1

            # find the current prediction, gradient
            p = dot(x_data_ptr, x_ind_ptr, weights_ptr, xnnz) + intercept
            gradient = loss._dloss(p, y)
            # gradient *= class_weight * sample_weight
            counter = 0
            for j in range(n_features):

                if counter < xnnz and x_ind_ptr[counter] == j:
                    val = x_data_ptr[counter]
                    counter += 1
                else:
                    val = 0.0
                full_gradient = (val * gradient +
                                 alpha * weights_ptr[j])
                sum_gradient[j] += (full_gradient -
                                    gradient_memory[current_index][j])
                gradient_memory[current_index][j] = full_gradient
                weights_ptr[j] -= eta * sum_gradient[j] / num_seen

            intercept -= eta * gradient

    free(gradient_memory)
    free(sum_gradient)
    # free(seen)

    return intercept

def fast_fit_sparse(SequentialDataset dataset,
                    np.ndarray[double, ndim=1, mode='c'] weights,
                    int n_samples,
                    int n_features,
                    int n_iter,
                    LossFunction loss,
                    double eta,
                    double alpha
                    ):

    cdef double* weights_ptr = &weights[0]
    cdef double *x_data_ptr
    cdef int *x_ind_ptr
    cdef double y
    cdef double sample_weight
    cdef int xnnz
    cdef int idx
    cdef int k
    cdef int current_index
    cdef int class_weight
    cdef int counter
    cdef double full_gradient

    # the total number of samples seen
    cdef double num_seen = 0.0
    # vector of booleans indicating whether this sample has been seen
    cdef np.ndarray[bint, ndim=1] seen_array = np.zeros(n_samples,
                                                        dtype=np.int32,
                                                        order="c")
    cdef bint* seen = <bint*> seen_array.data

    # the sum of gradients for each feature
    cdef np.ndarray[double, ndim=1] sum_gradient_array = \
        np.zeros(n_features,
                 dtype=np.double,
                 order="c")
    cdef double* sum_gradient = <double*> sum_gradient_array.data

    # the previously seen gradient for each sample
    cdef np.ndarray[double, ndim=1] gradient_memory_array = \
        np.zeros(n_samples,
                 dtype=np.double,
                 order="c")
    cdef double* gradient_memory = <double*> gradient_memory_array.data

    # the cumulative sums needed for JIT params
    cdef np.ndarray[double, ndim=1] cumulative_sums_array = \
        np.empty(n_samples * n_iter,
                 dtype=np.double,
                 order="c")
    cdef double* cumulative_sums = <double*> cumulative_sums_array.data

    # the index for the last time this feature was updated
    cdef np.ndarray[int, ndim=1] feature_hist_array = \
        np.zeros(n_features,
                 dtype=np.int32,
                 order="c")
    cdef int* feature_hist = <int*> feature_hist_array.data

    # the scalar used for multiplying z
    cdef double wscale = 1.0

    cumulative_sums[0] = 0.0

    cdef double intercept = 0.0

    with nogil:
        for k in range(n_iter * n_samples):
            # extract a random sample
            current_index = dataset.random(&x_data_ptr,
                                           &x_ind_ptr,
                                           &xnnz,
                                           &y,
                                           &sample_weight)

            # update the number of samples seen and the seen array
            if not seen[current_index]:
                num_seen += 1.0
                seen[current_index] = True

            # make the weight updates
            for j in range(xnnz):
                idx = x_ind_ptr[j]
                weights_ptr[idx] -= ((cumulative_sums[k] -
                                      cumulative_sums[feature_hist[idx]]) *
                                     sum_gradient[idx])
                feature_hist[idx] = k

            # find the current prediction, gradient
            p = (wscale * dot(x_data_ptr, x_ind_ptr,
                 weights_ptr, xnnz)) + intercept
            gradient = loss._dloss(p, y)

            # make the updates to the sum of gradients
            for j in range(xnnz):
                idx = x_ind_ptr[j]
                val = x_data_ptr[j]
                update = val * gradient
                sum_gradient[idx] += (update -
                                      gradient_memory[current_index] * val)

            gradient_memory[current_index] = gradient

            intercept -= eta * gradient

            wscale *= 1.0 - eta * alpha
            cumulative_sums[k + 1] = (cumulative_sums[k] +
                                      eta / (wscale * num_seen))

        k = n_samples * n_iter
        for j in range(n_features):
            weights_ptr[j] -= ((cumulative_sums[k] -
                                cumulative_sums[feature_hist[j]]) *
                               sum_gradient[j])
            weights_ptr[j] *= wscale
        wscale = 1.0

    return intercept


cdef double dot(double* x_data_ptr, int* x_ind_ptr, double* w_data_ptr,
                int xnnz) nogil:
        cdef int j
        cdef int idx
        cdef double innerprod = 0.0

        for j in range(xnnz):
            idx = x_ind_ptr[j]
            innerprod += w_data_ptr[idx] * x_data_ptr[j]
        return innerprod

