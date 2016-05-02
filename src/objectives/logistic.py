"""
Utils related to the logistic regression.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

def log1pexp(x):
    """log(1 + exp(x))"""
    return np.logaddexp(0, x)


def sigmoid(x):
  return 1.0 / log1pexp(-x)


def binary_logistic_loss(linear_o, y):
    """Returns a vector of logistic losses of each object.

    Given a vector of linear ouputs a vector of ground truth target values y
    returns logistic losses with respect to the linear outputs.
    Linear outputs can be e.g. <w, x_i> + b.
    """
    return log1pexp(-y.flatten() * linear_o.flatten()) / linear_o.size


def binary_logistic_loss_grad(linear_o, y):
    """Derivative of the binary_logistic_loss w.r.t. the linear output"""
    # Sometimes denom overflows, but it's OK, since if it's very large, it would
    # be set to INF and the output correctly takes the value of 0.
    # TODO: Fix overflow warnings.
    denom = 1 + np.exp(y.flatten() * linear_o.flatten())
    return -y / (denom * linear_o.size)


def _multinomial_loss(linear_o, y):
    raise NotImplementedError()


def preprocess(X, y, info=None):
    """Prepare the data for the learning"""
    if info is None:
        info = {}
        info['classes'] = np.unique(y)
        n_classes = info['classes'].size
        if n_classes < 2:
            raise ValueError("This solver needs samples of 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r." % info['classes'][0])

        if n_classes > 2:
             raise NotImplementedError("multiclass is not implemented yet.")

    idx_min_1 = (y == info['classes'][0])
    y = np.ones(y.shape)
    y[idx_min_1] = -1
    return X, y, info


def linear_init(X, y, fit_intercept=True):
    logreg = LogisticRegression(fit_intercept=fit_intercept)
    logreg.fit(X, y)
    if fit_intercept:
        intercept = logreg.intercept_[0]
    else:
        intercept = logreg.intercept_
    return logreg.coef_[0, :], intercept
