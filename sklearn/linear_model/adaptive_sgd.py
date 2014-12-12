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
    fit_method = {"standard": plain_sgd, "average": average_sgd}

    def __init__(self, loss, penalty='l2', alpha=0.0001, C=1.0,
                 l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False,
                 verbose=0, epsilon=0.1, random_state=None,
                 learning_rate="optimal", eta0=0.0, power_t=0.5,
                 warm_start=False, average=False):
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


class BaseAdaptiveSGDClassifier(six.with_metaclass(ABCMeta, BaseAdaptiveSGD,
                                                   BaseSGDClassifier,
                                                   LinearClassifierMixin)):
    def __init__(self, loss="hinge", penalty='l2', alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, n_iter=5, shuffle=False, verbose=0,
                 epsilon=0.1, n_jobs=1, random_state=None,
                 learning_rate="optimal", eta0=0.01, power_t=0.0,
                 class_weight=None, warm_start=False, average=False):

        self.n_jobs = n_jobs
        self.class_weight = class_weight

        super(BaseAdaptiveSGDClassifier, self).__init__(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept, n_iter=n_iter, shuffle=shuffle,
            verbose=verbose, epsilon=epsilon,
            random_state=random_state, learning_rate=learning_rate, eta0=eta0,
            power_t=power_t, warm_start=warm_start,
            average=average)


class BaseAdaptiveSGDRegressor(BaseAdaptiveSGD, BaseSGDRegressor,
                               RegressorMixin):

    def __init__(self, loss="squared_loss", penalty="l2", alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False,
                 verbose=0, epsilon=0.1, random_state=None,
                 learning_rate="invscaling", eta0=0.01, power_t=0.0,
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
            eta0=eta0, power_t=power_t,
            warm_start=warm_start,
            average=average)


class AdaptiveSGDClassifier(BaseAdaptiveSGDClassifier, _LearntSelectorMixin):
    loss_functions = {
        "hinge": (Hinge, 1.0),
    }

    def __init__(self, loss="hinge", penalty='l2', alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, n_iter=5, shuffle=False, verbose=0,
                 epsilon=0.1, n_jobs=1, random_state=None,
                 learning_rate="optimal", eta0=0.01, power_t=0.0,
                 class_weight=None, warm_start=False, average=False):
        super(AdaptiveSGDClassifier, self).__init__(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept, n_iter=n_iter, shuffle=shuffle,
            verbose=verbose, epsilon=epsilon, n_jobs=n_jobs,
            random_state=random_state, learning_rate=learning_rate, eta0=eta0,
            power_t=power_t, class_weight=class_weight, warm_start=warm_start,
            average=average)


class AdaptiveSGDRegressor(BaseAdaptiveSGDRegressor, _LearntSelectorMixin):
    loss_functions = {
        "log": (Log, 1.0),
    }

    def __init__(self, loss="squared_loss", penalty="l2", alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False,
                 verbose=0, epsilon=0.1, random_state=None,
                 learning_rate="invscaling", eta0=0.01, power_t=0.0,
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
            eta0=eta0, power_t=power_t,
            warm_start=warm_start,
            average=average)
