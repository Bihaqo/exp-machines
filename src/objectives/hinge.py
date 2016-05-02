"""
Utils related to the hinge loss (SVM).
"""

from sklearn.svm import LinearSVC
import numpy as np


def hinge_loss(linear_o, y):
    """Returns a vector of hinge losses of each object.

    Given a vector of linear outputs a vector of ground truth target values y
    returns hinge losses with respect to the linear outputs.
    Linear outputs can be e.g. <w, x_i> + b.
    """
    return np.maximum(0, 1 - y.flatten() * linear_o.flatten()) / float(linear_o.size)


def hinge_loss_grad(linear_o, y):
    """Derivative of the hinge_loss w.r.t. the linear output"""
    n = linear_o.size
    active_idx = (1 - y.flatten() * linear_o.flatten()) > 0
    grad = np.zeros(n)
    # TODO: check the gradient, optimization says "Desired error not necessarily achieved due to precision loss."
    grad[active_idx] = -y[active_idx] / float(n)
    return grad


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
    svm = LinearSVC(fit_intercept=fit_intercept)
    svm.fit(X, y)
    if fit_intercept:
        intercept = svm.intercept_[0]
    else:
        intercept = svm.intercept_
    return svm.coef_[0, :], intercept
