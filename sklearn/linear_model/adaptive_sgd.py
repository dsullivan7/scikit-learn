import numpy as np

from abc import ABCMeta

from .base import LinearClassifierMixin
from ..base import RegressorMixin
from ..feature_selection.from_model import _LearntSelectorMixin
from .sgd_fast_adaptive import plain_sgd, average_sgd
from .stochastic_gradient import BaseSGD, BaseSGDRegressor, BaseSGDClassifier
from ..utils import check_random_state
from ..externals import six
from .sgd_fast import Hinge
from .sgd_fast import Log


class BaseAdaptiveSGD(BaseSGD):
    def __init__(self, loss, penalty='l2', alpha=0.0001, C=1.0,
                 l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False,
                 verbose=0, epsilon=0.1, random_state=None,
                 learning_rate="optimal", eta0=0.0, eps0=0.1, power_t=0.5,
                 warm_start=False, average=False):
        self.eps0 = eps0
        super(BaseAdaptiveSGD, self).__init__(
            loss=loss, penalty=penalty,
            alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            n_iter=n_iter, shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0, power_t=power_t,
            warm_start=warm_start,
            average=average)

    def _fit_method(self, coef_init, intercept_init, average_coef_init,
                    average_intercept_init, loss_function,
                    penalty_type,
                    dataset, n_iter, seed,
                    pos_weight, neg_weight,
                    learning_rate_type, intercept_decay):

        intercepts = {}
        coefs = {}
        if not self.average > 0:
            standard_coef, standard_intercept = \
                plain_sgd(
                    coef_init, intercept_init,
                    loss_function,
                    penalty_type, self.alpha, self.C,
                    self.l1_ratio, dataset, n_iter,
                    int(self.fit_intercept), int(self.verbose),
                    int(self.shuffle), seed,
                    pos_weight, neg_weight,
                    learning_rate_type, self.eta0,
                    self.power_t, self.t_, self.eps0, intercept_decay)

            intercepts["standard"] = standard_intercept
            coefs["standard"] = standard_coef

        else:
            standard_coef, standard_intercept, average_coef, \
                average_intercept = \
                average_sgd(
                    coef_init, intercept_init, average_coef_init,
                    average_intercept_init, loss_function,
                    penalty_type, self.alpha, self.C,
                    self.l1_ratio, dataset, n_iter,
                    int(self.fit_intercept), int(self.verbose),
                    int(self.shuffle), seed,
                    pos_weight, neg_weight,
                    learning_rate_type, self.eta0,
                    self.power_t, self.t_, self.eps0, intercept_decay,
                    self.average)

            intercepts["standard"] = standard_intercept
            coefs["standard"] = standard_coef
            intercepts["average"] = average_intercept
            coefs["average"] = average_coef

        return coefs, intercepts


class BaseAdaptiveSGDClassifier(six.with_metaclass(ABCMeta, BaseAdaptiveSGD,
                                                   BaseSGDClassifier,
                                                   LinearClassifierMixin)):
    def __init__(self, loss="hinge", penalty='l2', alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, n_iter=5, shuffle=False, verbose=0,
                 epsilon=0.1, n_jobs=1, random_state=None,
                 learning_rate="optimal", eta0=0.01, eps0=0.1, power_t=0.0,
                 class_weight=None, warm_start=False, average=False):

        self.n_jobs = n_jobs
        self.class_weight = class_weight

        super(BaseAdaptiveSGDClassifier, self).__init__(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept, n_iter=n_iter, shuffle=shuffle,
            verbose=verbose, epsilon=epsilon,
            random_state=random_state, learning_rate=learning_rate, eta0=eta0,
            power_t=power_t, eps0=eps0, warm_start=warm_start,
            average=average)


class BaseAdaptiveSGDRegressor(BaseAdaptiveSGD, BaseSGDRegressor,
                               RegressorMixin):

    def __init__(self, loss="squared_loss", penalty="l2", alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False,
                 verbose=0, epsilon=0.1, random_state=None,
                 learning_rate="invscaling", eta0=0.01, eps0=0.1, power_t=0.0,
                 warm_start=False, average=False):
        super(BaseAdaptiveSGDRegressor, self).__init__(
            loss=loss, penalty=penalty,
            alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            n_iter=n_iter, shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0, eps0=eps0, power_t=power_t,
            warm_start=warm_start,
            average=average)


class AdaptiveSGDClassifier(BaseAdaptiveSGDClassifier, _LearntSelectorMixin):
    loss_functions = {
        "hinge": (Hinge, 1.0),
    }

    def __init__(self, loss="hinge", penalty='l2', alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, n_iter=5, shuffle=False, verbose=0,
                 epsilon=0.1, n_jobs=1, random_state=None,
                 learning_rate="optimal", eta0=0.01, eps0=0.1, power_t=0.0,
                 class_weight=None, warm_start=False, average=False):
        super(AdaptiveSGDClassifier, self).__init__(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept, n_iter=n_iter, shuffle=shuffle,
            verbose=verbose, epsilon=epsilon, n_jobs=n_jobs,
            random_state=random_state, learning_rate=learning_rate, eta0=eta0,
            eps0=eps0, power_t=power_t, class_weight=class_weight,
            warm_start=warm_start, average=average)


class AdaptiveSGDRegressor(BaseAdaptiveSGDRegressor, _LearntSelectorMixin):
    loss_functions = {
        "log": (Log, 1.0),
    }

    def __init__(self, loss="squared_loss", penalty="l2", alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False,
                 verbose=0, epsilon=0.1, random_state=None,
                 learning_rate="invscaling", eta0=0.01, eps0=0.1, power_t=0.0,
                 warm_start=False, average=False):
        super(AdaptiveSGDRegressor, self).__init__(
            loss=loss, penalty=penalty,
            alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            n_iter=n_iter, shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0, eps0=eps0, power_t=power_t,
            warm_start=warm_start,
            average=average)
