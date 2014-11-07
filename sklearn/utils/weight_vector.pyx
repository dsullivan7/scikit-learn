# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck <larsmans@gmail.com>
#         Danny Sullivan <dsullivan7@hotmail.com>
#
# Licence: BSD 3 clause

cimport cython
from libc.limits cimport INT_MAX
from libc.math cimport sqrt
import numpy as np
cimport numpy as np

cdef extern from "cblas.h":
    double ddot "cblas_ddot"(int, double *, int, double *, int) nogil
    void dscal "cblas_dscal"(int, double, double *, int) nogil
    void daxpy "cblas_daxpy" (int, double, const double*,
                              int, double*, int) nogil


np.import_array()


cdef class WeightVector(object):
    """Dense vector represented by a scalar and a numpy array.

    The class provides methods to ``add`` a sparse vector
    and scale the vector.
    Representing a vector explicitly as a scalar times a
    vector allows for efficient scaling operations.

    Attributes
    ----------
    w : ndarray, dtype=double, order='C'
        The numpy array which backs the weight vector.
    aw : ndarray, dtype=double, order='C'
        The numpy array which backs the average_weight vector.
    w_data_ptr : double*
        A pointer to the data of the numpy array.
    wscale : double
        The scale of the vector.
    n_features : int
        The number of features (= dimensionality of ``w``).
    sq_norm : double
        The squared norm of ``w``.
    """

    def __cinit__(self,
                  np.ndarray[double, ndim=1, mode='c'] w,
                  np.ndarray[double, ndim=1, mode='c'] aw,
                  np.ndarray[double, ndim=1, mode='c'] wscale_vector):
        cdef double *wdata = <double *>w.data

        if w.shape[0] > INT_MAX:
            raise ValueError("More than %d features not supported; got %d."
                             % (INT_MAX, w.shape[0]))
        self.w = w
        self.w_data_ptr = wdata
        self.wscale = 1.0
        self.wscale_ptr = <double *>wscale_vector.data
        self.n_etas = wscale_vector.shape[0]
        self.n_features = w.shape[0]
        self.sq_norm = ddot(<int>w.shape[0], wdata, 1, wdata, 1)

        self.aw = aw
        if self.aw is not None:
            self.aw_data_ptr = <double *>aw.data
            self.average_a = 0.0
            self.average_b = 1.0

    cdef void add(self, double *x_data_ptr, int *x_ind_ptr, int xnnz,
                  double *eta_ptr, double update) nogil:
        """Scales sample x by constant c and adds it to the weight vector.

        This operation updates ``sq_norm``.

        Parameters
        ----------
        x_data_ptr : double*
            The array which holds the feature values of ``x``.
        x_ind_ptr : np.intc*
            The array which holds the feature indices of ``x``.
        xnnz : int
            The number of non-zero features of ``x``.
        c : double
            The scaling constant for the example.
        """
        cdef int j
        cdef int idx
        cdef double val
        cdef double innerprod = 0.0
        cdef double xsqnorm = 0.0
        cdef int eta_index

        # the next two lines save a factor of 2!
        cdef double wscale = self.wscale
        cdef double* w_data_ptr = self.w_data_ptr
        cdef double* wscale_ptr = self.wscale_ptr

        for j in range(xnnz):
            idx = x_ind_ptr[j]
            val = x_data_ptr[j]
            if self.n_etas > 1:
                eta_index = idx
            else:
                eta_index = 0
            innerprod += (w_data_ptr[idx] * val)
            xsqnorm += (val * val)
            w_data_ptr[idx] += val * (eta_ptr[eta_index] * update / wscale_ptr[eta_index])

        self.sq_norm += (xsqnorm * eta_ptr[0] * eta_ptr[0]) + (2.0 * innerprod * wscale_ptr[0] * eta_ptr[0])

    # Update the average weights according to the sparse trick defined
    # here: http://research.microsoft.com/pubs/192769/tricks-2012.pdf
    # by Leon Bottou
    cdef void add_average(self, double *x_data_ptr, int *x_ind_ptr, int xnnz,
                          double* eta_ptr, double update, double num_iter) nogil:
        """Updates the average weight vector.

        Parameters
        ----------
        x_data_ptr : double*
            The array which holds the feature values of ``x``.
        x_ind_ptr : np.intc*
            The array which holds the feature indices of ``x``.
        xnnz : int
            The number of non-zero features of ``x``.
        c : double
            The scaling constant for the example.
        num_iter : double
            The total number of iterations.
        """
        cdef int j
        cdef int idx
        cdef double val
        cdef double mu = 1.0 / num_iter
        cdef int eta_index
        cdef double average_a = self.average_a
        cdef double wscale = self.wscale
        cdef double* aw_data_ptr = self.aw_data_ptr
        cdef double* wscale_ptr = self.wscale_ptr

        for j in range(xnnz):
            idx = x_ind_ptr[j]
            val = x_data_ptr[j]
            if self.n_etas > 1:
                eta_index = idx
            else:
                eta_index = 0
            aw_data_ptr[idx] += (self.average_a * val * (eta_ptr[eta_index] * update / wscale_ptr[eta_index]))

        # Once the the sample has been processed
        # update the average_a and average_b
        if num_iter > 1:
            self.average_b /= (1.0 - mu)
        self.average_a += mu * self.average_b * wscale_ptr[0]

    cdef double dot(self, double *x_data_ptr, int *x_ind_ptr,
                    int xnnz) nogil:
        """Computes the dot product of a sample x and the weight vector.

        Parameters
        ----------
        x_data_ptr : double*
            The array which holds the feature values of ``x``.
        x_ind_ptr : np.intc*
            The array which holds the feature indices of ``x``.
        xnnz : int
            The number of non-zero features of ``x`` (length of x_ind_ptr).

        Returns
        -------
        innerprod : double
            The inner product of ``x`` and ``w``.
        """
        cdef int j
        cdef int idx
        cdef double innerprod = 0.0
        cdef double* w_data_ptr = self.w_data_ptr
        cdef double* wscale_ptr = self.wscale_ptr
        cdef double wscale = self.wscale
        cdef int eta_index
        for j in range(xnnz):
            idx = x_ind_ptr[j]
            if self.n_etas > 1:
                eta_index = idx
            else:
                eta_index = 0
            innerprod += wscale_ptr[eta_index] * w_data_ptr[idx] * x_data_ptr[j]
        return innerprod

    cdef void scale(self, double c) nogil:
        """Scales the weight vector by a constant ``c``.

        It updates ``wscale`` and ``sq_norm``. If ``wscale`` gets too
        small we call ``reset_swcale``."""
        self.wscale *= c
        self.sq_norm *= (c * c)
        if self.wscale < 1e-9:
            self.reset_wscale()

    cdef void scale_vector(self, double l1_ratio,
                           double* eta_ptr, double alpha) nogil:
        if alpha == 0.0:
            return
        for j in range(self.n_etas):
            self.wscale_ptr[j] *= (1.0 - ((1.0 - l1_ratio) *
                                   eta_ptr[j] * alpha))
        cdef double norm = (1.0 - ((1.0 - l1_ratio) *
                            eta_ptr[0] * alpha))
        self.sq_norm *= (norm * norm)

    cdef void reset_wscale(self) nogil:
        """Scales each coef of ``w`` by ``wscale`` and resets it to 1. """
        cdef double* wscale_ptr = self.wscale_ptr

        if self.aw is not None:
            daxpy(<int>self.aw.shape[0], self.average_a,
                  <double *>self.w.data, 1, <double *>self.aw.data, 1)
            dscal(<int>self.aw.shape[0], 1.0 / self.average_b,
                  <double *>self.aw.data, 1)
            self.average_a = 0.0
            self.average_b = 1.0

        if self.n_etas > 1:
            for j in range(self.n_features):
                self.w_data_ptr[j] *= wscale_ptr[j]
                self.wscale_ptr[j] = 1.0
        else:
            dscal(<int>self.w.shape[0], self.wscale_ptr[0],
                  <double *>self.w.data, 1)
            # wscale_ptr[0] = 1.0

        self.wscale = 1.0

    cdef double norm(self) nogil:
        """The L2 norm of the weight vector. """
        return sqrt(self.sq_norm)
