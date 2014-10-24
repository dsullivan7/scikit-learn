cdef extern from "math.h":
    double pow(double, double) nogil
    double fmin(double, double) nogil
    double sqrt(double) nogil

cimport cython

cdef class LearningRate:
    cdef double eta(self, double eta0, double alpha, double t, double power_t):
        pass
    cdef double update(self, double gradient, double loss, double eta,
                       double norm, double C, double p, double y,
                       int is_hinge):
        return -eta * gradient

cdef class Constant(LearningRate):
    cdef double eta(self, double eta0, double alpha, double t, double power_t):
        return eta0

cdef class Optimal(LearningRate):
    @cython.cdivision(True)
    cdef double eta(self, double eta0, double alpha, double t, double power_t):
        return 1.0 / (alpha * (t - 1))

cdef class InvScaling(LearningRate):
    @cython.cdivision(True)
    cdef double eta(self, double eta0, double alpha, double t, double power_t):
        return eta0 / pow(t, power_t)

cdef class AdaGrad(LearningRate):

    def __cinit__(self):
        self.sum_squared_grad = 0.0
        self.eps0 = 1.E-7

    cdef double eta(self, double eta0, double alpha, double t, double power_t):
        return eta0

    @cython.cdivision(True)
    cdef double update(self, double gradient, double loss, double eta,
                       double norm, double C, double p, double y,
                       int is_hinge):
        self.sum_squared_grad += gradient ** 2.0 + self.eps0
        return -(eta / sqrt(self.sum_squared_grad)) * gradient

cdef class AdaDelta(LearningRate):
    def __cinit__(self):
        self.sum_squared_grad = 0
        self.rho0 = 0.95
        # self.eps0 = 1.E-7
        self.eps0 = .1
        self.accugrad = 0
        self.accudelta = 0

    cdef double eta(self, double eta0, double alpha, double t, double power_t):
        return eta0

    @cython.cdivision(True)
    cdef double update(self, double gradient, double loss, double eta,
                       double norm, double C, double p, double y,
                       int is_hinge):
        agrad = self.rho0 * self.accugrad + \
            (1. - self.rho0) * gradient * gradient
        dx = - sqrt((self.accudelta + self.eps0) /
                    (agrad + self.eps0)) * gradient
        self.accudelta = self.rho0 * self.accudelta +\
            (1. - self.rho0) * dx * dx
        self.accugrad = agrad
        return dx

cdef class PA(LearningRate):
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
    cdef double eta(self, double eta0, double alpha, double t, double power_t):
        return eta0

    @cython.cdivision(True)
    cdef double update(self, double gradient, double loss, double eta,
                       double norm, double C, double p, double y,
                       int is_hinge):
        update = loss / norm
        update = fmin(C, update)
        update *= self._get_multiplier(is_hinge, p, y)
        return update

cdef class PA2(PA):
    cdef double eta(self, double eta0, double alpha, double t, double power_t):
        return eta0

    @cython.cdivision(True)
    cdef double update(self, double gradient, double loss, double eta,
                       double norm, double C, double p, double y,
                       int is_hinge):
        update = loss / (norm + 0.5 / C)
        update *= self._get_multiplier(is_hinge, p, y)
        return update
