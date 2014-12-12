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

    def _proxy_partial_fit(self, X, y, coef_init, intercept_init,
                           average_coef_init, average_intercept_init,
                           loss, learning_rate, n_iter, pos_weight, neg_weight,
                           sample_weight):
        """Fit a X and y"""
        n_samples, n_features = X.shape

        # if average is not true, average_coef, and average_intercept will be
        # unused
        if not self.average:
            if coef_init is None:
                coef_init = np.zeros(n_features, dtype=np.float64, order="C")
            if intercept_init is None:
                intercept_init = 0.0
        else:
            if coef_init is None:
                coef_init = np.zeros(n_features, dtype=np.float64, order="C")
            if intercept_init is None:
                intercept_init = 0.0
            if average_coef_init is None:
                coef_init = np.zeros(n_features, dtype=np.float64, order="C")
            if intercept_init is None:
                average_intercept_init = 0.0

        if self.t_ is None:
            self.t_ = 0.0

        assert y.shape[0] == sample_weight.shape[0]
        dataset, intercept_decay = self._make_dataset(X, y, sample_weight)

        penalty_type = self._get_penalty_type(self.penalty)
        learning_rate_type = self._get_learning_rate_type(learning_rate)
        loss_function = self._get_loss_function(loss)

        # XXX should have random_state_!
        random_state = check_random_state(self.random_state)
        # numpy mtrand expects a C long which is a signed 32 bit integer under
        # Windows
        seed = random_state.randint(0, np.iinfo(np.int32).max)

        intercepts = {}
        coefs = {}
        if not self.average > 0:
            standard_coef, standard_intercept = \
                plain_sgd(coef_init, intercept_init, loss_function,
                          penalty_type, self.alpha, self.C, self.l1_ratio,
                          dataset, n_iter, int(self.fit_intercept),
                          int(self.verbose), int(self.shuffle), seed,
                          pos_weight, neg_weight,
                          learning_rate_type, self.eta0,
                          self.power_t, self.t_, intercept_decay)

            intercepts["standard"] = standard_intercept
            coefs["standard"] = standard_coef

        else:
            standard_coef, standard_intercept, average_coef, \
                average_intercept = \
                average_sgd(coef_init, intercept_init, average_coef_init,
                            average_intercept_init,
                            loss_function, penalty_type,
                            self.alpha, self.C, self.l1_ratio, dataset,
                            n_iter, int(self.fit_intercept),
                            int(self.verbose), int(self.shuffle),
                            seed, pos_weight, neg_weight,
                            learning_rate_type, self.eta0,
                            self.power_t, self.t_,
                            intercept_decay,
                            self.average)

            intercepts["standard"] = standard_intercept
            coefs["standard"] = standard_coef
            intercepts["average"] = average_intercept
            coefs["average"] = average_coef

        return coefs, intercepts


class BaseAdaptiveSGDClassifier(six.with_metaclass(ABCMeta, BaseAdaptiveSGD,
                                                   BaseSGDClassifier,
                                                   LinearClassifierMixin)):
    pass


class BaseAdaptiveSGDRegressor(BaseAdaptiveSGD, BaseSGDRegressor,
                               RegressorMixin):

    pass


class AdaptiveSGDClassifier(BaseAdaptiveSGDClassifier, _LearntSelectorMixin):
    loss_functions = {
        "hinge": (Hinge, 1.0)
    }


class AdaptiveSGDRegressor(BaseAdaptiveSGDRegressor, _LearntSelectorMixin):
    loss_functions = {
        "log": (Log, 1.0),
    }
