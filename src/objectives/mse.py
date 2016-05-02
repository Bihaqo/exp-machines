"""
Utils related to the least squared regression.
"""

import numpy as np
from sklearn.linear_model import LinearRegression


def mse_loss(linear_o, y):
    """Returns a vector of mean squared errors of each object.

    Given a vector of linear ouputs a vector of ground truth target values y
    returns squared residuals.
    Linear outputs can be e.g. <w, x_i> + b.
    """
    return 0.5 * (linear_o.flatten() - y.flatten())**2 / linear_o.size


def mse_loss_grad(linear_o, y):
    """Derivative of the mse_loss w.r.t. the linear output"""
    return (linear_o.flatten() - y.flatten()) / linear_o.size


def preprocess(X, y, info=None):
    """Prepare the data for the learning"""
    # No preprocessing.
    return X, y, info

def linear_init(X, y, fit_intercept=True):
    regr = LinearRegression(fit_intercept=fit_intercept)
    regr.fit(X, y)
    return regr.coef_, regr.intercept_
