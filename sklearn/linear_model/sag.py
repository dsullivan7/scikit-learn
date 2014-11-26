import numpy as np
import scipy.sparse as sp

from abc import ABCMeta, abstractmethod

from .base import LinearClassifierMixin, LinearModel, SparseCoefMixin
from ..base import RegressorMixin, BaseEstimator
from sklearn.feature_selection.from_model import _LearntSelectorMixin
from ..utils import check_X_y, compute_class_weight, check_random_state
from ..utils.seq_dataset import ArrayDataset, CSRDataset
from ..externals import six
from ..externals.joblib import Parallel, delayed
from .sgd_fast import Log, SquaredLoss
from .sag_fast import sag, sag_sparse, get_auto_eta

MAX_INT = np.iinfo(np.int32).max

"""For sparse data intercept updates are scaled by this decay factor to avoid
intercept oscillation."""
SPARSE_INTERCEPT_DECAY = 0.01


# taken from http://stackoverflow.com/questions/1816958
# useful for passing instance methods to Parallel
def multiprocess_method(instance, name, args=()):
    "indirect caller for instance methods and multiprocessing"
    return getattr(instance, name)(*args)


# The inspiration for SAG comes from:
# "Minimizing Finite Sums with the Stochastic Average Gradient" by
# Mark Schmidt, Nicolas Le Roux, Francis Bach. 2013. <hal-00860051>
#
# https://hal.inria.fr/hal-00860051/PDF/sag_journal.pdf
class BaseSAG(six.with_metaclass(ABCMeta, SparseCoefMixin)):
    def __init__(self, alpha=0.0001, fit_intercept=True, n_iter=5, verbose=0,
                 random_state=None, eta0='auto', warm_start=False):
        self.gradient_memory = None
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.n_iter = n_iter
        self.verbose = verbose
        self.eta0 = eta0
        self.random_state = random_state
        self.warm_start = warm_start

        self.coef_ = None
        self.intercept_ = None

        self.num_seen_ = None
        self.seen_ = None
        self.sum_gradient_ = None
        self.gradient_memory_ = None

    def _fit(self, X, y, coef_init=None, intercept_init=None,
             sample_weight=None, sum_gradient_init=None,
             gradient_memory_init=None, seen_init=None, num_seen_init=None,
             weight_pos=1.0, weight_neg=1.0):

        n_samples, n_features = X.shape[0], X.shape[1]

        # initialize all parameters if there is no init
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float64, order='C')

        if intercept_init is None:
            intercept_init = 0.0

        if coef_init is None:
            coef_init = np.zeros(n_features, dtype=np.float64, order='C')

        if sum_gradient_init is None:
            sum_gradient_init = np.zeros(n_features, dtype=np.float64,
                                         order='C')

        if gradient_memory_init is None:
            gradient_memory_init = np.zeros(n_samples, dtype=np.float64,
                                            order='C')

        if seen_init is None:
            seen_init = np.zeros(n_samples, dtype=np.int32, order='C')

        if num_seen_init is None:
            num_seen_init = 0

        random_state = check_random_state(self.random_state)

        # check which type of Sequential Dataset is needed
        if sp.issparse(X):
            dataset = CSRDataset(X.data, X.indptr, X.indices,
                                 y, sample_weight,
                                 seed=random_state.randint(MAX_INT))
            intercept_decay = SPARSE_INTERCEPT_DECAY
        else:
            dataset = ArrayDataset(X, y, sample_weight,
                                   seed=random_state.randint(MAX_INT))
            intercept_decay = 1.0

        # set the eta0 if needed, 'auto' is 1 / 4L where L is the max sum of
        # squares for over all samples
        if self.eta0 == 'auto':
            step_size = get_auto_eta(dataset, self.alpha, n_samples,
                                     self.loss_function)
        else:
            step_size = self.eta0

        # intercept_ = sag(dataset, coef_init.ravel(), n_samples, n_features,
        #                  self.n_iter, self.loss_function,
        #                  step_size, self.alpha)
        # num_seen = 0

        intercept_, num_seen = sag_sparse(dataset, coef_init.ravel(),
                                          intercept_init, n_samples,
                                          n_features, self.n_iter,
                                          self.loss_function,
                                          step_size, self.alpha,
                                          sum_gradient_init.ravel(),
                                          gradient_memory_init.ravel(),
                                          seen_init.ravel(),
                                          num_seen_init, weight_pos,
                                          weight_neg,
                                          intercept_decay)

        return (coef_init.reshape(1, -1), intercept_,
                sum_gradient_init.reshape(1, -1),
                gradient_memory_init.reshape(1, -1),
                seen_init.reshape(1, -1),
                num_seen)


class BaseSAGClassifier(six.with_metaclass(ABCMeta, BaseSAG)):
    @abstractmethod
    def __init__(self, alpha=0.0001,
                 fit_intercept=True, n_iter=5, verbose=0,
                 n_jobs=1, random_state=None,
                 eta0='auto', class_weight=None, warm_start=False):
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.loss_function = Log()
        super(BaseSAGClassifier, self).__init__(alpha=alpha,
                                                fit_intercept=fit_intercept,
                                                n_iter=n_iter,
                                                verbose=verbose,
                                                random_state=random_state,
                                                eta0=eta0,
                                                warm_start=warm_start)

    def _fit(self, X, y, coef_init=None, intercept_init=None,
             sample_weight=None, sum_gradient_init=None,
             gradient_memory_init=None, seen_init=None,
             num_seen_init=None):
        X, y = check_X_y(X, y, "csr", copy=False, order='C',
                         dtype=np.float64)
        n_samples, n_features = X.shape[0], X.shape[1]

        self.classes_ = np.unique(y)
        self.expanded_class_weight_ = compute_class_weight(self.class_weight,
                                                           self.classes_, y)

        if self.classes_.shape[0] <= 1:
            # there is only one class
            raise ValueError("The number of class labels must be "
                             "greater than one.")
        elif self.classes_.shape[0] == 2:
            # binary classifier
            coef, intercept, sum_gradient, gradient_memory, seen, num_seen = \
                self._fit_target_class(X, y, self.classes_[1],
                                       coef_init, intercept_init,
                                       sample_weight, sum_gradient_init,
                                       gradient_memory_init,
                                       seen_init, num_seen_init)
        else:
            # multiclass classifier
            coef = []
            intercept = []
            sum_gradient = []
            gradient_memory = []
            seen = []
            num_seen = []

            # perform a fit for all classes, one verse all
            results = Parallel(n_jobs=self.n_jobs,
                               backend="threading",
                               verbose=self.verbose)(
                # we have to use a call to multiprocess_method instead of the
                # plain instance method because pickle will not work on
                # instance methods in python 2.6 and 2.7
                delayed(multiprocess_method)(self, "_fit_target_class",
                                             (X, y, cl,
                                              coef_init, intercept_init,
                                              sample_weight,
                                              sum_gradient_init,
                                              gradient_memory_init,
                                              seen_init, num_seen_init))
                for cl in self.classes_)

            # append results to the correct array
            for (coef_cl, intercept_cl, sum_gradient_cl, gradient_memory_cl,
                 seen_cl, num_seen_cl) in results:
                coef.append(coef_cl)
                intercept.append(intercept_cl)
                sum_gradient.append(sum_gradient_cl)
                gradient_memory.append(gradient_memory_cl)
                seen.append(seen_cl)
                num_seen.append(num_seen_cl)

            # stack all arrays to transform into np arrays
            coef = np.vstack(coef)
            intercept = np.array(intercept)
            sum_gradient = np.vstack(sum_gradient)
            gradient_memory = np.vstack(gradient_memory)
            seen = np.vstack(seen)
            num_seen = np.array(num_seen)

        self.coef_ = coef
        self.intercept_ = intercept
        self.sum_gradient_ = sum_gradient
        self.gradient_memory_ = gradient_memory
        self.seen_ = seen
        self.num_seen_ = num_seen

    def _fit_target_class(self, X, y, target_class, coef_init=None,
                          intercept_init=None, sample_weight=None,
                          sum_gradient_init=None, gradient_memory_init=None,
                          seen_init=None, num_seen_init=None):
        if self.classes_.shape[0] == 2:
            if self.warm_start:
                # init parameters for binary classifier
                coef_init = self.coef_
                intercept_init = self.intercept_
                sum_gradient_init = self.sum_gradient_
                gradient_memory_init = self.gradient_memory_
                seen_init = self.seen_
                num_seen_init = self.num_seen_

            weight_pos = self.expanded_class_weight_[1]
            weight_neg = self.expanded_class_weight_[0]
        else:
            class_index = np.where(self.classes_ == target_class)[0][0]
            if self.warm_start:
                # init parameters for multi-class classifier
                if self.coef_ is not None:
                    coef_init = self.coef_[class_index]
                if self.intercept_ is not None:
                    intercept_init = self.intercept_[class_index]
                if self.sum_gradient_ is not None:
                    sum_gradient_init = self.sum_gradient_[class_index]
                if self.gradient_memory_ is not None:
                    gradient_memory_init = self.gradient_memory_[class_index]
                if self.seen_ is not None:
                    seen_init = self.seen_[class_index]
                if self.num_seen_ is not None:
                    num_seen_init = self.num_seen_[class_index]

            weight_pos = self.expanded_class_weight_[class_index]
            weight_neg = 1.0

        n_samples, n_features = X.shape[0], X.shape[1]

        y_encoded = np.ones(n_samples)
        y_encoded[y != target_class] = -1.0

        return super(BaseSAGClassifier, self)._fit(X, y_encoded,
                                                   coef_init, intercept_init,
                                                   sample_weight,
                                                   sum_gradient_init,
                                                   gradient_memory_init,
                                                   seen_init, num_seen_init,
                                                   weight_pos, weight_neg)


class BaseSAGRegressor(six.with_metaclass(ABCMeta, BaseSAG)):
    @abstractmethod
    def __init__(self, alpha=0.0001, fit_intercept=True, n_iter=5, verbose=0,
                 random_state=None, eta0='auto', warm_start=False):

        self.loss_function = SquaredLoss()
        super(BaseSAGRegressor, self).__init__(alpha=alpha,
                                               fit_intercept=fit_intercept,
                                               n_iter=n_iter,
                                               verbose=verbose,
                                               random_state=random_state,
                                               eta0=eta0,
                                               warm_start=warm_start)

    def _fit(self, X, y, coef_init=None, intercept_init=None,
             sample_weight=None, sum_gradient_init=None,
             gradient_memory_init=None, seen_init=None,
             num_seen_init=None):
        X, y = check_X_y(X, y, "csr", copy=False, order='C', dtype=np.float64)
        y = y.astype(np.float64)

        if self.warm_start:
            coef_init = self.coef_
            intercept_init = self.intercept_
            sum_gradient_init = self.sum_gradient_
            gradient_memory_init = self.gradient_memory_
            seen_init = self.seen_
            num_seen_init = self.num_seen_

        (self.coef_, self.intercept_, self.sum_gradient_,
         self.gradient_memory_, self.seen_, self.num_seen_) = \
            super(BaseSAGRegressor, self)._fit(X, y, coef_init,
                                               intercept_init,
                                               sample_weight,
                                               sum_gradient_init,
                                               gradient_memory_init,
                                               seen_init, num_seen_init)


class SAGClassifier(BaseSAGClassifier, _LearntSelectorMixin,
                    LinearClassifierMixin, BaseEstimator):
    """Linear classifiers (SVM, logistic regression, a.o.) with SAG training.

    This estimator implements regularized linear models with stochastic
    average gradient (SAG) learning: the gradient of the loss is estimated
    using a random sample from the dataset. The weights are then updated
    according to the sum of gradients seen thus far divided by the number of
    unique samples seen. The inspiration for SAG comes from "Minimizing Finite
    Sums with the Stochastic Average Gradient" by Mark Schmidt, Nicolas Le
    Roux, and Francis Bach. 2013. <hal-00860051>
    https://hal.inria.fr/hal-00860051/PDF/sag_journal.pdf

    This implementation works with data represented as dense or sparse arrays
    of floating point values for the features. It will fit the data according
    to log loss.

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using either the squared euclidean norm
    L2.

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the regularization term. Defaults to 0.0001

    fit_intercept: bool, optional
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered. Defaults to True.

    n_iter: int, optional
        The number of passes over the training data (aka epochs). The number
        of iterations is set to 1 if using partial_fit.
        Defaults to 5.

    random_state: int or numpy.random.RandomState, optional
        The random_state of the pseudo random number generator to use when
        sampling the data.

    verbose: integer, optional
        The verbosity level

    n_jobs: integer, optional
        The number of CPUs to use to do the OVA (One Versus All, for
        multi-class problems) computation. -1 means 'all CPUs'. Defaults
        to 1.

    eta0 : double or "auto"
        The initial learning rate. The default value is 0.001.

    class_weight : dict, {class_label : weight} or "auto" or None, optional
        Preset for the class_weight fit parameter.

        Weights associated with classes. If not given, all classes
        are supposed to have weight one.

        The "auto" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies.

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.


    Attributes
    ----------
    coef_ : array, shape (1, n_features) if n_classes == 2 else (n_classes,
    n_features)
        Weights assigned to the features.

    intercept_ : array, shape (1,) if n_classes == 2 else (n_classes,)
        Constants in decision function.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import linear_model
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> Y = np.array([1, 1, 2, 2])
    >>> clf = linear_model.SAGClassifier()
    >>> clf.fit(X, Y)
    ... #doctest: +NORMALIZE_WHITESPACE
    SAGClassifier(alpha=0.0001, class_weight=None,
                  eta0='auto', fit_intercept=True,
                  n_iter=5, n_jobs=1, random_state=None,
                  verbose=0, warm_start=False)
    >>> print(clf.predict([[-0.8, -1]]))
    [1]

    See also
    --------
    SGDClassifier, LinearSVC, LogisticRegression, Perceptron

    """
    def __init__(self, alpha=0.0001, fit_intercept=True, n_iter=5,
                 verbose=0, n_jobs=1, random_state=None,
                 eta0='auto', class_weight=None, warm_start=False):

        super(SAGClassifier, self).__init__(alpha=alpha,
                                            class_weight=class_weight,
                                            fit_intercept=fit_intercept,
                                            n_iter=n_iter,
                                            verbose=verbose,
                                            n_jobs=n_jobs,
                                            random_state=random_state,
                                            eta0=eta0,
                                            warm_start=warm_start)

    def fit(self, X, y, coef_init=None, intercept_init=None,
            sample_weight=None, sum_gradient_init=None,
            gradient_memory_init=None, seen_init=None,
            num_seen_init=None):
        super(SAGClassifier, self)._fit(X, y, coef_init,
                                        intercept_init,
                                        sample_weight,
                                        sum_gradient_init,
                                        gradient_memory_init,
                                        seen_init, num_seen_init)
        return self


class SAGRegressor(BaseSAGRegressor, _LearntSelectorMixin,
                   LinearModel, RegressorMixin, BaseEstimator):
    """Linear model fitted by minimizing a regularized empirical loss with SAG

    SAG stands for Stochastic Average Gradient: the gradient of the loss is
    estimated each sample at a time and the model is updated along the way with
    a constant learning rate. The inspiration for SAG comes from "Minimizing
    Finite Sums with the Stochastic Average Gradient" by Mark Schmidt,
    Nicolas Le Roux, and Francis Bach. 2013. <hal-00860051>
    https://hal.inria.fr/hal-00860051/PDF/sag_journal.pdf

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using the squared euclidean norm
    L2.

    This implementation works with data represented as dense or sparse numpy
    arrays of floating point values for the features.

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the regularization term. Defaults to 0.0001

    fit_intercept: bool, optional
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered. Defaults to True.

    n_iter : int, optional
        The number of passes over the training data (aka epochs).
        Defaults to 5.

    random_state: int or numpy.random.RandomState, optional
        The random_state of the pseudo random number generator to use when
        sampling the data.

    verbose: integer, optional
        The verbosity level.

    eta0 : double or "auto"
        The initial learning rate [default 0.01].

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Weights asigned to the features.

    intercept_ : array, shape (1,)
        The intercept term.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import linear_model
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> y = np.random.randn(n_samples)
    >>> X = np.random.randn(n_samples, n_features)
    >>> clf = linear_model.SAGRegressor()
    >>> clf.fit(X, y)
    ... #doctest: +NORMALIZE_WHITESPACE
    SAGRegressor(alpha=0.0001, eta0='auto',
                 fit_intercept=True, n_iter=5, random_state=None,
                 verbose=0, warm_start=False)

    See also
    --------
    SGDRegressor, Ridge, ElasticNet, Lasso, SVR

    """
    def __init__(self, alpha=0.0001, fit_intercept=True, n_iter=5, verbose=0,
                 random_state=None, eta0='auto', warm_start=False):
        super(SAGRegressor, self).__init__(alpha=alpha,
                                           fit_intercept=fit_intercept,
                                           n_iter=n_iter,
                                           verbose=verbose,
                                           random_state=random_state,
                                           eta0=eta0, warm_start=warm_start)

    def fit(self, X, y, coef_init=None, intercept_init=None,
            sample_weight=None, sum_gradient_init=None,
            gradient_memory_init=None, seen_init=None,
            num_seen_init=None):
        super(SAGRegressor, self)._fit(X, y, coef_init,
                                       intercept_init,
                                       sample_weight,
                                       sum_gradient_init,
                                       gradient_memory_init,
                                       seen_init, num_seen_init)
        return self
