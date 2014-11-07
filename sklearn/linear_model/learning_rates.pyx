cdef extern from "math.h":
    double pow(double, double) nogil
    double fmin(double, double) nogil
    double sqrt(double) nogil
import numpy as np
cimport cython
cimport numpy as np
from sklearn.utils.weight_vector cimport WeightVector

cdef class LearningRate:
    cdef void eta(self, double *eta_ptr, double eta0, double alpha, double t,
                  double power_t, double gradient, int n_features,
                  double* x_data_ptr, int* x_ind_ptr, int xnnz,
                  WeightVector w):
        eta_ptr[0] = eta0
    cdef double update(self, double gradient, double loss,
                       double norm, double C, double p, double y,
                       int is_hinge):
        return gradient

cdef class Constant(LearningRate):

    def __reduce__(self):
        return Constant, ()

cdef class Optimal(LearningRate):
    @cython.cdivision(True)
    cdef void eta(self, double *eta_ptr, double eta0, double alpha, double t,
                  double power_t, double gradient, int n_features,
                  double* x_data_ptr, int* x_ind_ptr, int xnnz,
                  WeightVector w):
        eta_ptr[0] = 1.0 / (alpha * (t - 1))

    def __reduce__(self):
        return Optimal, ()

cdef class InvScaling(LearningRate):
    @cython.cdivision(True)
    cdef void eta(self, double *eta_ptr, double eta0, double alpha, double t,
                  double power_t, double gradient, int n_features,
                  double* x_data_ptr, int* x_ind_ptr, int xnnz,
                  WeightVector w):
        eta_ptr[0] = eta0 / pow(t, power_t)

    def __reduce__(self):
        return InvScaling, ()

cdef class Adaptive(LearningRate):
    cdef double _compute_eta(self, double full_gradient,
                             double val, int idx, double eta0, int n_features):
        pass

    @cython.cdivision(True)
    cdef void eta(self, double *eta_ptr, double eta0, double alpha, double t,
                  double power_t, double gradient, int n_features,
                  double* x_data_ptr, int* x_ind_ptr, int xnnz,
                  WeightVector w):

        cdef int counter = 0
        cdef int j = 0
        cdef double val

        if alpha == 0.0:
            # we can use a sparse trick here
            for j in range(xnnz):
                idx = x_ind_ptr[j]
                val = x_data_ptr[j]
                full_gradient = gradient * val
                eta_ptr[idx] = self._compute_eta(full_gradient, val, idx,
                                                 eta0, w.n_features)
        else:
            # since alpha is non-zero, we have to update
            # all sum_squared_gradients
            for j in range(w.n_features):
                if counter < xnnz and x_ind_ptr[counter] == j:
                    val = x_data_ptr[counter]
                    counter += 1
                else:
                    val = 0.0

                full_gradient = (gradient * val +
                                 alpha * w.w_data_ptr[j] * w.wscale_ptr[j])
                eta_ptr[j] = self._compute_eta(full_gradient, val, j,
                                               eta0, w.n_features)

cdef class AdaGrad(Adaptive):

    def __cinit__(self, double sum_squared_grad, double eps0, double rho0):
        self.sum_squared_grad = sum_squared_grad
        self.sum_squared_grad_vector = None
        self.eps0 = eps0

    @cython.cdivision(True)
    cdef double _compute_eta(self, double full_gradient,
                             double val, int idx, double eta0, int n_features):
        if self.sum_squared_grad_vector is None:
            self.sum_squared_grad_vector = \
                np.zeros((n_features,), dtype=np.float64, order="c")

        cdef double eps0 = self.eps0
        cdef double* sum_squared_grad_vector_ptr = \
            <double *>self.sum_squared_grad_vector.data
        sum_squared_grad_vector_ptr[idx] += full_gradient * full_gradient

        return eta0 / sqrt(sum_squared_grad_vector_ptr[idx] + eps0)

    def __reduce__(self):
        return AdaGrad, (self.sum_squared_grad, self.eps0, self.rho0)


cdef class AdaDelta(Adaptive):
    def __cinit__(self, double sum_squared_grad, double eps0, double rho0):
        self.rho0 = rho0
        self.eps0 = eps0
        self.accugrad = None
        self.accudelta = None

    @cython.cdivision(True)
    cdef double _compute_eta(self, double full_gradient,
                             double val, int idx, double eta0, int n_features):
        if self.accugrad is None:
            self.accugrad = \
                np.zeros((n_features,), dtype=np.float64, order="c")
        if self.accudelta is None:
            self.accudelta = \
                np.zeros((n_features,), dtype=np.float64, order="c")

        cdef double eps0 = self.eps0
        cdef double rho0 = self.rho0
        cdef double* accugrad = <double *>self.accugrad.data
        cdef double* accudelta = <double *>self.accudelta.data

        accugrad[idx] *= rho0
        accugrad[idx] += (1. - rho0) * full_gradient * full_gradient
        dx = sqrt((accudelta[idx] + eps0) / (accugrad[idx] + eps0))
        accudelta[idx] *= rho0
        accudelta[idx] += (1. - rho0) * dx * dx

        return dx

    def __reduce__(self):
        return AdaDelta, (self.sum_squared_grad, self.eps0, self.rho0)

cdef class PA(LearningRate):
    cdef void eta(self, double *eta_ptr, double eta0, double alpha, double t,
                  double power_t, double gradient, int n_features,
                  double* x_data_ptr, int* x_ind_ptr, int xnnz,
                  WeightVector w):
        eta_ptr[0] = -1.0

    cdef double _get_multiplier(self, int is_hinge, double p, double y):
        if is_hinge:
            # classification
            return y
        elif y - p < 0.0:
            # regression
            return -1.0
        else:
            return 1.0

cdef class PA1(PA):

    @cython.cdivision(True)
    cdef double update(self, double gradient, double loss,
                       double norm, double C, double p, double y,
                       int is_hinge):
        update = loss / norm
        update = fmin(C, update)
        update *= self._get_multiplier(is_hinge, p, y)
        return update

    def __reduce__(self):
        return PA1, ()

cdef class PA2(PA):

    @cython.cdivision(True)
    cdef double update(self, double gradient, double loss,
                       double norm, double C, double p, double y,
                       int is_hinge):
        update = loss / (norm + 0.5 / C)
        update *= self._get_multiplier(is_hinge, p, y)
        return update

    def __reduce__(self):
        return PA2, ()
