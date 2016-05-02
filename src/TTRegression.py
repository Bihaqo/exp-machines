from sklearn.linear_model.base import BaseEstimator, LinearClassifierMixin
import sklearn
import numpy as np
from copy import deepcopy
from utils import roc_auc_score_reversed

import tt

import logging


class TTRegression(BaseEstimator, LinearClassifierMixin):
    """This class alows to optimize functions of the following structure:
        sum_i f(<w, g(x_i)> + b, y_i) + lambda <w, w> / 2
       where the sum is over the objects in the dataset, w is a tensor in the TT-format.
    Parameters
    ----------
    tt_model : {'all-subsets'}
    loss_name : {'logistic', 'hinge', 'mse'}
    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
    max_iter : int, default: 100
    solver : {'riemannian-sgd', 'sgd'}
        Algorithm to use in the optimization problem.
    batch_size : Positive integer OR -1
        -1 is for the full gradient, that is using the whole training set in
        each batch.
    reg : float, default: 0
        L2 regularization coefficient.
        WARNING: reg parameter means different things for different solvers.
        Riemannian-sgd assumes L2 regularization in terms of the tensor w:
            reg * <w, w> / 2
        while sgd solver assumes regularization in terms of the cores elements:
            reg * <w.core, w.core> / 2
    verbose : int
        Set verbose to any positive number for verbosity.
    Attributes
    ----------
    coef_ : TT-tensor
    intercept_ : real
        Intercept (a.k.a. bias) added to the decision function.
        If `fit_intercept` is set to False, the intercept is set to zero.
    logger : instance of the class Logging
        Contains all the logged details (e.g. loss on each iteration).
    """

    def __init__(self, tt_model, loss_name, rank,
                 solver='riemannian-sgd', batch_size=-1, fit_intercept=True,
                 reg=0., exp_reg=1.0, dropout=None, max_iter=100, verbose=0,
                 persuit_init=False, coef0=None, intercept0=None):

        # Save all the params as class attributes. It's required by the
        # BaseEstimator class.
        self.tt_model = tt_model
        self.loss_name = loss_name
        self.rank = rank
        self.solver = solver
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.reg = reg
        self.exp_reg = exp_reg
        self.dropout = dropout
        self.max_iter = max_iter
        self.verbose = verbose
        self.persuit_init = persuit_init
        self.coef0 = coef0
        self.intercept0 = intercept0

    def parse_params(self):
        """Checks the parameters and sets class attributes according to them.

        This have to be done on each call to fit(), since parameters can change
        after __init__ via BaseEstimator.set_params method.
        """
        if self.reg < 0:
            raise ValueError("Regularization should be positive.")
        if self.exp_reg < 1.:
            raise ValueError("Exponential regularization should be greater than 1.0")
        if np.abs(self.reg) < 1e-12 and np.abs(self.exp_reg - 1.) > 1e-12:
            print('WARNING: exp_reg has no effect without reg. '
                  'The regularization works like this:\n'
                  'reg * (W_00^2 + exp_reg * W_01^2 + exp_reg * W_10^2 + '
                  'exp_reg^2 * W_11^2)')
        self.watched_metrics = {}
        if self.tt_model == 'all-subsets':
            import models.all_subsets as all_subsets
            self.tt_dot = all_subsets.vectorized_tt_dot
            self.project = all_subsets.project_all_subsets
            self.tensorize_linear_init = all_subsets.tensorize_linear_init
            self.gradient_wrt_cores = all_subsets.gradient_wrt_cores
            self.object_tensor = all_subsets.subset_tensor
        else:
            raise ValueError("Only all-subsets model is supported.")

        if self.loss_name == 'logistic':
            import objectives.logistic as logistic
            self.loss = logistic.binary_logistic_loss
            self.loss_grad = logistic.binary_logistic_loss_grad
            self.preprocess = logistic.preprocess
            self.linear_init = logistic.linear_init
            self.watched_metrics = {
                "logistic": self.loss,
                "auc": roc_auc_score_reversed
            }
        elif self.loss_name == 'mse':
            # TODO: MSE loss fluctuates instead of steadily improving. Debug!
            # The possible reason is the lack of regularization (norm of w
            # goes to 1e10 and machine errors become too large).
            import objectives.mse as mse
            self.loss = mse.mse_loss
            self.loss_grad = mse.mse_loss_grad
            self.preprocess = mse.preprocess
            self.linear_init = mse.linear_init
            self.watched_metrics = {
                "mse": self.loss
            }
        elif self.loss_name == 'hinge':
            from objectives import hinge
            self.loss = hinge.hinge_loss
            self.loss_grad = hinge.hinge_loss_grad
            self.preprocess = hinge.preprocess
            self.linear_init = hinge.linear_init
            self.watched_metrics = {
                "hinge": self.loss,
                "auc": roc_auc_score_reversed
            }

        else:
            raise ValueError("Only logistic, mse and hinge losses are supported.")

    def fit(self, X_, y_):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training object-feature matrix, where n_samples in the number
            of samples and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.
        Returns
        -------
        self : object
            Returns self.
        """
        self.fit_log_val(X_, y_)

    def fit_log_val(self, X_, y_, val_X_=None, val_y_=None):
        """Fit the model according to the given training data. Log validation loss on each epoch.
        Parameters
        ----------
        X_ : {array-like}, shape (n_samples, n_features)
            Training object-feature matrix, where n_samples in the number
            of samples and n_features is the number of features.
        y_ : array-like, shape (n_samples,)
             Target vector relative to X_.
        val_X_ : {array-like}, shape (n_val_samples, n_features)
                 Validation object-feature matrix.
        val_y_ : array-like, shape (n_val_samples,)
                 Target vector relative to val_X_.
        Returns
        -------
        self : object
            Returns self.
        """
        self.parse_params()

        self.logger = logging.Logging(self.verbose, self.watched_metrics, log_w_norm=True)
        if np.abs(self.reg) > 1e-10 and self.logger.disp():
            print('WARNING: reg parameter means different things for different solvers.\n'
                  'Riemannian-sgd assumes L2 regularization in terms of the tensor w:\n'
                  '\treg * <w, w> / 2\n'
                  'while sgd solver assumes regularization in terms of the cores elements:\n'
                  '\treg * <w.core, w.core> / 2\n')

        if self.persuit_init and self.coef0 is not None:
            if self.logger.disp():
                print('WARNING: persuit_init parameter is not compatible with '
                      'explicitly providing initial values.')

        # TODO: deal with sparse data.
        # Copy the dataset, since preprocessing changes user's data, which is messy.
        X = deepcopy(X_)
        y = deepcopy(y_)
        X, y, self.info = self.preprocess(X, y)
        if val_X_ is not None and val_y_ is not None:
            val_X = deepcopy(val_X_)
            val_y = deepcopy(val_y_)
            val_X, val_y, self.info = self.preprocess(val_X, val_y, self.info)
        else:
            val_X, val_y = None, None
        if self.coef0 is None:
            self.coef_, self.intercept_ = self.linear_init(X, y)
            # Convert coefficients of linear model into the TT-format.
            self.coef_ = self.tensorize_linear_init(self.coef_, self.intercept_)
            if self.rank < max(self.coef_.r):
                # Decrease rank if necessary.
                self.coef_ = self.coef_.round(eps=0, rmax=self.rank)
            # Once we incorporated the intercept in coef, we don't need to have it
            # separately.
            self.intercept_ = 0
            # Increase the rank up to the desired one.
            if self.persuit_init and self.solver != 'riemannian-sgd':
                self.persuit_init = False
                if self.logger.disp():
                    print('WARNING: persuit_init is supported only by the riemannian-sgd solver')
            if self.persuit_init:
                if self.solver == 'riemannian-sgd':
                    from optimizers.riemannian_sgd import increase_rank
                    self.coef_ = increase_rank(self.coef_, self.rank, X, y,
                                               self.tt_dot, self.loss, self.loss_grad,
                                               self.project, self.object_tensor,
                                               self.reg)
            else:
                n = self.coef_.n
                for _ in range(self.rank - max(self.coef_.r)):
                    self.coef_ = self.coef_ + 0 * tt.ones(n)
                self.coef_ = self.coef_.round(eps=0)
            assert(max(self.coef_.r) == self.rank)
        else:
            self.coef_ = self.coef0
            self.intercept_ = self.intercept0
            if self.intercept_ is None:
                self.intercept_ = 0

        if self.solver == 'riemannian-sgd':
            from optimizers.riemannian_sgd import riemannian_sgd
            w, b = riemannian_sgd(X, y, self.tt_dot, self.loss, self.loss_grad,
                                  self.project, w0=self.coef_,
                                  intercept0=self.intercept_,
                                  fit_intercept=self.fit_intercept,
                                  val_x=val_X, val_y=val_y,
                                  reg=self.reg, exp_reg=self.exp_reg,
                                  dropout=self.dropout,
                                  batch_size=self.batch_size,
                                  num_passes=self.max_iter,
                                  logger=self.logger, verbose_period=1,
                                  beta=0.5, rho=0.1)
            self.coef_, self.intercept_ = w, b
        elif self.solver == 'sgd':
            if self.dropout is not None:
                print('WARNING: dropout for "sgd" solver is not supported.')

            from optimizers.core_sgd import core_sgd
            w, b = core_sgd(X, y, self.tt_dot, self.loss, self.loss_grad,
                            self.gradient_wrt_cores, w0=self.coef_,
                            intercept0=self.intercept_,
                            fit_intercept=self.fit_intercept,
                            val_x=val_X, val_y=val_y, reg=self.reg,
                            batch_size=self.batch_size,
                            num_passes=self.max_iter,
                            logger=self.logger, verbose_period=1,
                            beta=0.5, rho=0.1)
            self.coef_, self.intercept_ = w, b
        else:
            raise ValueError("Only 'riemannian-sgd' and 'sgd' solvers are supported.")
        return self

    def decision_function(self, X):
        """Returns linear output of the model.
        Returns <w, g(x_i)> + b for all objects x_i, where w is in the TT-format.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : array-like, shape = [n_samples, 1]
        """
        if not hasattr(self, "coef_"):
            raise sklearn.exceptions.NotFittedError("Call fit before prediction")
        return self.tt_dot(self.coef_, X) + self.intercept_

    def predict_proba(self, X):
        """Probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        if self.loss_name == 'logistic':
            return super(TTRegression, self)._predict_proba_lr(X)
        else:
            raise RuntimeError("Can't compute probabilities, the model was "
                               "fitted with non-logistic loss.")

    def predict_log_proba(self, X):
        """Log of probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))
