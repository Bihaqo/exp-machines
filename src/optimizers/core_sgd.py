import numpy as np
from scipy.optimize import minimize_scalar
import tt


# TODO: add Adam scheme support.
def core_sgd(train_x, train_y, vectorized_tt_dot_h, loss_h,
             loss_grad_h, grad_wrt_cores_h, w0, intercept0=0,
             fit_intercept=True, val_x=None, val_y=None, reg=0,
             batch_size=-1, num_passes=30, seed=None,
             logger=None, verbose_period=1,
             debug=False,
             beta=0.5, rho=0.1):
    """SGD w.r.t. TT-cores optimization for a linear model with weights in TT.

    The objective function is
        reg <w, w> + \sum_i f(d(w, x_i) + b, y_i)
        * where f(o, y) is the loss w.r.t. one object (this function is from R^2 to R);
        * d(w, x_i) is the dot product between the tensor w and the tensor build
            from the vector x_i.
    """
    num_objects = train_x.shape[0]
    is_val_set_provided = False
    if val_x is not None and val_y is not None:
        is_val_set_provided = True

    if batch_size == -1:
        # Full gradient learning.
        batch_size = num_objects
    # TODO: correctly process the last batch.
    num_batches = num_objects // batch_size

    if seed is not None:
        np.random.seed(seed)

    w = w0
    b = b = intercept0
    # TODO: start not from zero in case we are resuming the learning.
    start_epoch = 0

    if logger is not None:
        logger.before_first_iter(train_x, train_y, w,  lambda w, x: vectorized_tt_dot_h(w, x) + b, num_passes, num_objects)

    # TODO: use some more resanable step initialization (e.g. a line search on the first iteration).
    step_w = 1
    for e in xrange(start_epoch, num_passes):
        idx_perm = np.random.permutation(num_objects)
        for batch_idx in xrange(num_batches):
            # Allow the step to grow.
            step_w *= 1.2
            start = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size
            curr_idx = idx_perm[start:end]
            batch_y = train_y[curr_idx]
            batch_w_x = vectorized_tt_dot_h(w, train_x[curr_idx, :])
            batch_linear_o = batch_w_x + b
            batch_loss_arr = loss_h(batch_linear_o, batch_y)
            wcore_wcore = w.core.dot(w.core)
            batch_loss = np.sum(batch_loss_arr) + reg * wcore_wcore / 2.0
            batch_grad_coef = loss_grad_h(batch_linear_o, batch_y)
            w_cores = tt.tensor.to_list(w)
            gradient = grad_wrt_cores_h(w_cores, train_x[curr_idx, :], batch_grad_coef)
            gradient += reg * w.core

#           Armiho step choosing.
            step_prev_w = step_w
            gradient_norm = np.linalg.norm(gradient)
            while step_w > 1e-10:
                new_w = w.copy()
                new_w.core += -step_w * gradient
                new_w_x = vectorized_tt_dot_h(new_w, train_x[curr_idx, :])

                if fit_intercept:
                    b_objective = lambda b: np.sum(loss_h(new_w_x + b, batch_y))
                    m = minimize_scalar(b_objective)
                    b = m.x
                    new_loss = m.fun
                else:
                    new_loss = np.sum(loss_h(new_w_x + b, batch_y))
                new_loss += reg * new_w.core.dot(new_w.core) / 2.0
                if new_loss <= batch_loss - rho * step_w * gradient_norm**2:
                    break
                step_w *= beta
            w = new_w

        if (logger is not None) and e % verbose_period == 0:
            logger.after_each_iter(e, train_x, train_y, w, lambda w, x: vectorized_tt_dot_h(w, x) + b, stage='train')
            if is_val_set_provided:
                logger.after_each_iter(e, val_x, val_y, w, lambda w, x: vectorized_tt_dot_h(w, x) + b, stage='valid')

    return w, b
