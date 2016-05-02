import numpy as np
from scipy.optimize import minimize_scalar, fmin_ncg, fmin_bfgs
import tt
from tt.riemannian import riemannian


def _regularized_loss_step(steps, loss_h, y, w_x, dir_x, b, gradb, reg, w_w,
                           w_dir, grdir):
    new_w_x = w_x - steps[0] * dir_x
    new_b = b - steps[1] * gradb
    loss = loss_h(new_w_x + new_b, y)
    neww_neww = (w_w + steps[0]**2 * grdir - 2 * steps[0] * w_dir)
    return np.sum(loss) + reg * neww_neww / 2


def _regularized_loss_step_grad(steps, loss_grad_h, y, w_x, dir_x, b, gradb,
                                reg, w_dir, grdir):
    new_w_x = w_x - steps[0] * dir_x
    new_b = b - steps[1] * gradb
    loss_grad = loss_grad_h(new_w_x + new_b, y)
    grad = np.zeros(2)
    grad[0] = -loss_grad.dot(dir_x) + reg * (steps[0] * grdir - w_dir)
    grad[1] = -np.sum(loss_grad) * gradb
    return grad


def increase_rank(w0, rank, train_x, train_y, vectorized_tt_dot_h,
                  loss_h, loss_grad_h, project_h, object_tensor_h, reg):
    """Implements the idea from the paper
    Riemannian Pursuit for Big Matrix Recovery

    That is, to init the tensor with the desired rank, we add orthogonal
    component of the current gradient to our current low-rank estimate w0.
    """
    w = w0
    if rank > max(w.r):
        # Choose not too many objects, so that the rank is reasonable.
        num_objects_used = rank * 5
        w_x = vectorized_tt_dot_h(w, train_x[:num_objects_used, :])
        grad_coef = loss_grad_h(w_x, train_y[:num_objects_used])
        proj_grad = project_h(w, train_x[:num_objects_used, :], grad_coef, reg=reg)
        grad = reg * w
        for i in range(0, num_objects_used):
            grad = grad + grad_coef[i] * object_tensor_h(train_x[i, :])
        orth = grad - proj_grad
        orth = orth.round(eps=0, rmax=rank-max(w0.r))

        batch_w_x = vectorized_tt_dot_h(w, train_x)
        w_w = w.norm()**2
        batch_orth_x = vectorized_tt_dot_h(orth, train_x)
        orth_orth = orth.norm()**2
        w_orth = tt.dot(w, orth)

        def w_step_objective(w_step):
            steps = np.array([w_step, 0])
            obj = _regularized_loss_step(steps, loss_h, train_y,
                                         batch_w_x, batch_orth_x, 0,
                                         0, reg, w_w,
                                         w_orth, orth_orth)
            return obj
        step_w = minimize_scalar(w_step_objective).x
        w = (w - step_w * orth).round(eps=0)
    return w


def build_reg_tens(n, exp_reg):
    reg_tens = tt.ones(n)
    reg_tens.core[1::2] = exp_reg
    return reg_tens


def riemannian_sgd(train_x, train_y, vectorized_tt_dot_h, loss_h,
                   loss_grad_h, project_h, w0, intercept0=0,
                   fit_intercept=True, val_x=None, val_y=None,
                   reg=0., exp_reg=1., dropout=None, batch_size=-1,
                   num_passes=30, seed=None, logger=None, verbose_period=1,
                   debug=False, beta=0.5, rho=0.1):
    """Riemannian SGD method optimization for a linear model with weights in TT.

    The objective function is
        reg <w, w> + \sum_i f(d(w, x_i) + b, y_i)
        * where f(o, y) is the loss w.r.t. one object (this function is from R^2 to R);
        * d(w, x_i) is the dot product between the tensor w and the tensor build
            from the vector x_i.
    """
    num_objects, num_features = train_x.shape
    is_val_set_provided = False
    if val_x is not None and val_y is not None:
        is_val_set_provided = True

    if seed is not None:
        np.random.seed(seed)

    if batch_size == -1:
        # Full gradient learning.
        batch_size = num_objects
    # TODO: correctly process the last batch.
    num_batches = num_objects // batch_size

    w = w0
    b = intercept0
    # TODO: start not from zero in case we are resuming the learning.
    start_epoch = 0

    if logger is not None:
        logger.before_first_iter(train_x, train_y, w, lambda w, x: vectorized_tt_dot_h(w, x) + b, num_passes, num_objects)

    reg_tens = build_reg_tens(w.n, exp_reg)

    for e in xrange(start_epoch, num_passes):
        idx_perm = np.random.permutation(num_objects)
        for batch_idx in xrange(num_batches):
            start = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size
            curr_idx = idx_perm[start:end]
            curr_batch = train_x[curr_idx, :]
            if dropout is not None:
                dropout_mask = np.random.binomial(1, dropout,
                                                  size=curr_batch.shape)
                # To make the expected value of <W, dropout(X)> equals to <W, X>.
                dropout_mask = dropout_mask / dropout
                curr_batch = dropout_mask * curr_batch
            batch_y = train_y[curr_idx]
            batch_w_x = vectorized_tt_dot_h(w, curr_batch)
            batch_linear_o = batch_w_x + b
            batch_loss_arr = loss_h(batch_linear_o, batch_y)
            wreg = w * reg_tens
            wregreg = w * reg_tens * reg_tens
            wreg_wreg = wreg.norm()**2
            batch_loss = np.sum(batch_loss_arr) + reg * wreg_wreg / 2.0
            batch_grad_coef = loss_grad_h(batch_linear_o, batch_y)
            batch_gradient_b = np.sum(batch_grad_coef)
            direction = project_h(w, curr_batch, batch_grad_coef, reg=0)
            direction = riemannian.project(w, [direction, reg * wregreg])
            batch_dir_x = vectorized_tt_dot_h(direction, curr_batch)

            dir_dir = direction.norm()**2
            wreg_dir = tt.dot(wreg, direction)
            if fit_intercept:
                # TODO: Use classical Newton-Raphson (with hessian).
                step_objective = lambda s: _regularized_loss_step(s, loss_h, batch_y, batch_w_x, batch_dir_x, b, batch_gradient_b, reg, wreg_wreg, wreg_dir, dir_dir)
                step_gradient = lambda s: _regularized_loss_step_grad(s, loss_grad_h, batch_y, batch_w_x, batch_dir_x, b, batch_gradient_b, reg, wreg_dir, dir_dir)
                step0_w, step0_b = fmin_bfgs(step_objective, np.ones(2), fprime=step_gradient, gtol=1e-10, disp=logger.disp())
            else:
                def w_step_objective(w_step):
                    steps = np.array([w_step, 0])
                    obj = _regularized_loss_step(steps, loss_h, batch_y,
                                                 batch_w_x, batch_dir_x, b,
                                                 batch_gradient_b, reg, wreg_wreg,
                                                 wreg_dir, dir_dir)
                    return obj
                step0_w = minimize_scalar(w_step_objective).x


            # TODO: consider using Probabilistic Line Searches for Stochastic Optimization.
#           Armiho step choosing.
            step_w = step0_w
            # <gradient, direction> =
            # = <(\sum_i coef[i] * x_i + reg * w), direction> =
            # = \sum_i coef[i] <x_i, direction> + reg * <w, direction>
            grad_times_direction = batch_dir_x.dot(batch_grad_coef) + reg * wreg_dir
            while step_w > 1e-10:
                new_w = (w - step_w * direction).round(eps=0, rmax=max(w.r))
                new_w_x = vectorized_tt_dot_h(new_w, curr_batch)

                if fit_intercept:
                    b_objective = lambda b: np.sum(loss_h(new_w_x + b, batch_y))
                    m = minimize_scalar(b_objective)
                    b = m.x
                    new_loss = m.fun
                else:
                    new_loss = np.sum(loss_h(new_w_x + b, batch_y))
                new_wreg = new_w * reg_tens
                new_loss += reg * new_wreg.norm()**2 / 2.0
                if new_loss <= batch_loss - rho * step_w * grad_times_direction:
                    break
                step_w *= beta
            w = new_w

        if (logger is not None) and e % verbose_period == 0:
            logger.after_each_iter(e, train_x, train_y, w, lambda w, x: vectorized_tt_dot_h(w, x) + b, stage='train')
            if is_val_set_provided:
                logger.after_each_iter(e, val_x, val_y, w, lambda w, x: vectorized_tt_dot_h(w, x) + b, stage='valid')

    return w, b
