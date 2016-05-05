import tt
import numpy as np
from tt.riemannian import riemannian
from numba import jit


def reshape(a, sz):
    return np.reshape(a, sz, order="F")


def subset_tensor(x):
    tens = tt.ones(2 * np.ones(len(x)))
    tens.core[range(1, 2*len(x), 2)] = x
    return tens


def tensorize_linear_init(coef, intercept):
    """Initialize all-subset model with weights of a linear model.

    Given a vector of d coefficients of a linear model returns a d-dimensional
    TT tensor of all-subsets model which represent the same linear model.
    The resulting TT-tensor ranks equal 2."""
    coef = np.array(coef)
    num_features = coef.shape[0]
    w_cores = [None] * num_features
    for i in range(num_features):
        if i == 0:
            core = np.array([[[1, 0], [0, coef[i]]]])
        elif i == num_features-1:
            core = np.array([[[intercept], [coef[i]]], [[1], [0]]])
        else:
            core = np.zeros([2, 2, 2])
            core[:, 0, :] = np.eye(2)
            core[:, 1, :] = np.array([[0, 1], [0, 0]]) * coef[i]
        w_cores[i] = core
    return tt.tensor.from_list(w_cores)


def gradient_wrt_cores(w_cores, x, dfdz):
    """Compute the gradient of the loss function w.r.t. the cores of w.

    Compute the gradient of
        \sum_i f(<w, subset_tensor(x_i)>)
    w.r.t. to the cores of w.

    w_cores is a list of cores of w.
    x is an object-feature matrix.
    dfdz is a vector of derivatives of the function f().
    """
    num_objects, num_features = x.shape
    summed_cores = [None] * num_features
    for dim in xrange(num_features):
        r1, n, r2 = w_cores[dim].shape
        summed_cores[dim] = reshape(w_cores[dim][:, 1, :], (r1, r2, 1)) * x[np.newaxis, np.newaxis, :, dim]
        summed_cores[dim] += reshape(w_cores[dim][:, 0, :], (r1, r2, 1))

    rhs = [None] * (num_features + 1)
    rhs[num_features] = np.ones((1, num_objects))
    for dim in xrange(num_features-1, 0, -1):
        rhs[dim] = np.einsum("ijk,jk->ik", summed_cores[dim], rhs[dim+1])
    lhs = np.ones((num_objects, 1))
    grad = [None] * num_features
    for dim in xrange(num_features):
        r1, n, r2 = w_cores[dim].shape
        grad[dim] = np.zeros((r1, n, r2))
        grad[dim][:, 0, :] = np.einsum("ij,ki,i->jk", lhs, rhs[dim+1], dfdz)
        grad[dim][:, 1, :] = np.einsum("ij,ki,i->jk", lhs, rhs[dim+1], dfdz * x[:, dim])
        lhs = np.einsum("ij,jki->ik", lhs, summed_cores[dim])
    return tt.tensor.from_list(grad).core


def project_all_subsets(w, X, coef=None, reg=0, debug=False):
    """ Project all the all_subset tensors X on the tangent space of tensor w.

    w is a tensor in the TT format.
    X is an object-feature matrix.
    The function computes the projection of the sum off all the subset tensors
    plus the w itself times the regularization coefficient:
        project(w, X) = P_w(reg * w + \sum_i subset_tensor(X[i, :]))
    ).
    This function implements an algorithm from the paper [1], theorem 3.1.
    This code is basically a copy-paste from the tt.riemannian.project
    with a few modifications.

    Returns a tensor in the TT format with the TT-ranks equal 2 * rank(w).
    """

    num_objects = X.shape[0]
    numDims, modeSize = w.d, w.n
    cores_w = tt.tensor.to_list(w)


    ############################################
    #### Specific to the all-subsets tensors ###
    ############################################
    # zCoresDim[k][i, :, :, :] is the k-th core of the i-th object.
    zCoresDim = [None] * numDims
    for dim in xrange(numDims):
        zCoresDim[dim] = np.zeros([num_objects, 1, 2, 1])
        zCoresDim[dim][:, 0, 0, 0] = 1
        zCoresDim[dim][:, 0, 1, 0] = X[:, dim]
    if coef is not None:
        zCoresDim[0][:, 0, :, 0] *= coef[:, np.newaxis]
    ############################################
    #################### End ###################
    ############################################
    # Initialize the cores of the projection_w(sum x[i]).
    coresP = []
    for dim in xrange(numDims):
        r1 = 2 * w.r[dim]
        r2 = 2 * w.r[dim+1]
        if dim == 0:
            r1 = 1
        if dim == numDims - 1:
            r2 = 1
        coresP.append(np.zeros((r1, modeSize[dim], r2)))
    # rhs[dim] is a num_objects x 1 x w.rank_dim.rank_dim ndarray.
    # Right to left orthogonalization of X and preparation of the rhs vectors.
    for dim in xrange(numDims-1, 0, -1):
        # Right to left orthogonalization of the X cores.
        cores_w = riemannian.cores_orthogonalization_step(cores_w, dim, left_to_right=False)
        r1, n, r2 = cores_w[dim].shape

        # Fill the right orthogonal part of the projection.
        coresP[dim][0:r1, :, 0:r2] = cores_w[dim]
    ############################################
    ########## L2 regularization term ##########
    ############################################
    r1, n, r2 = cores_w[0].shape
    coresP[0][:, :, 0:r2] = reg * cores_w[0]
    ############################################
    #################### End ###################
    ############################################

    rhs = [None] * (numDims+1)
    for dim in xrange(numDims):
        rhs[dim] = np.zeros([num_objects, 1, cores_w[dim].shape[0]])
    rhs[numDims] = np.ones([num_objects, 1, 1])

    for dim in xrange(numDims-1, 0, -1):
        riemannian._update_rhs(rhs[dim+1], cores_w[dim], zCoresDim[dim], rhs[dim])

    if debug:
        assert(np.allclose(w.full(), tt.tensor.from_list(cores_w).full()))

    # lsh is a num_objects x X.rank_dim x 1 ndarray.
    lhs = np.ones([num_objects, 1, 1])
    # Left to right sweep.
    for dim in xrange(numDims):
        cc = cores_w[dim].copy()
        r1, n, r2 = cc.shape
        if dim < numDims-1:
            # Left to right orthogonalization.
            cc = reshape(cc, (-1, r2))
            cc, rr = np.linalg.qr(cc)
            r2 = cc.shape[1]
            # Warning: since ranks can change here, do not use X.r!
            # Use cores_w[dim].shape instead.
            if debug:
                # Need to do it before the move non orthogonal part rr to
                # the cores_w[dim+1].
                rightQ = riemannian.right(tt.tensor.from_list(cores_w), dim+1)
            cores_w[dim] = reshape(cc, (r1, n, r2)).copy()
            cores_w[dim+1] = np.tensordot(rr, cores_w[dim+1], 1)

            new_lhs = np.zeros([num_objects, r2, 1])
            riemannian._update_lhs(lhs, cores_w[dim], zCoresDim[dim], new_lhs)

            # See the correspondic section in the non-jit version of this
            # code for a less confusing implementation of
            # the transformation below.
            currPCore = np.einsum('ijk,iklm->ijlm', lhs, zCoresDim[dim])
            currPCore = reshape(currPCore, (num_objects, r1*n, -1))
            currPCore -= np.einsum('ij,kjl->kil', cc, new_lhs)
            currPCore = np.einsum('ijk,ikl', currPCore, rhs[dim+1])
            currPCore = reshape(currPCore, (r1, modeSize[dim], r2))
            if dim == 0:
                coresP[dim][0:r1, :, 0:r2] += currPCore
            else:
                coresP[dim][r1:, :, 0:r2] += currPCore
            if debug:
                explicit_sum = np.zeros((r1, modeSize[dim], r2))
                for idx in xrange(num_objects):
                    leftQm1 = riemannian.left(tt.tensor.from_list(cores_w), dim-1)
                    leftQ = riemannian.left(tt.tensor.from_list(cores_w), dim)

                    obj_tensor = subset_tensor(X[idx, :])
                    first = np.tensordot(leftQm1.T, riemannian.unfolding(obj_tensor, dim-1), 1)
                    second = reshape(first, (-1, np.prod(modeSize[dim+1:])))
                    if dim < numDims-1:
                        explicit = second.dot(rightQ)
                        orth_cc = reshape(cores_w[dim], (-1, cores_w[dim].shape[2]))
                        explicit -= orth_cc.dot(leftQ.T.dot(riemannian.unfolding(obj_tensor, dim)).dot(rightQ))
                    else:
                        explicit = second
                    explicit_sum += reshape(explicit, currPCore.shape)
                assert(np.allclose(explicit_sum, currPCore))
            lhs = new_lhs

            if dim == 0:
                coresP[dim][0:r1, :, r2:] = cores_w[dim]
            else:
                coresP[dim][r1:, :, r2:] = cores_w[dim]

        if dim == numDims-1:
            coresP[dim][r1:, :, 0:r2] += np.einsum('ijk,iklm->jlm', lhs, zCoresDim[dim])

    if debug:
        assert(np.allclose(w.full(), tt.tensor.from_list(cores_w).full()))
    return tt.tensor.from_list(coresP)


# TODO: test it.
def _prepare_linear_core(w):
    """This function prepare the data to be used in _vectorized_tt_dot_jit"""
    res = np.zeros(w.core.shape[0])
    idx = 0
    num_dims = w.d
    cores_w = tt.tensor.to_list(w)
    r1, n, r2 = cores_w[0].shape
    for alpha_2 in xrange(r2):
        res[idx] = cores_w[0][0, 0, alpha_2]
        idx += 1
        res[idx] = cores_w[0][0, 1, alpha_2]
        idx += 1
    for dim in xrange(1, num_dims-1):
        r1, n, r2 = cores_w[dim].shape
        for alpha_2 in xrange(r2):
            for alpha_1 in xrange(r1):
                res[idx] = cores_w[dim][alpha_1, 0, alpha_2]
                idx += 1
                res[idx] = cores_w[dim][alpha_1, 1, alpha_2]
                idx += 1
    dim = num_dims-1
    r1, n, r2 = cores_w[dim].shape
    for alpha_1 in xrange(r1):
        res[idx] = cores_w[dim][alpha_1, 0, 0]
        idx += 1
        res[idx] = cores_w[dim][alpha_1, 1, 0]
        idx += 1
    return res


@jit(nopython=True)
def _vectorized_tt_dot_jit(linear_core_w, X, result, num_dims, ranks):
    """Compute a dot products between a tensor w and subset tensors built from x.

    In most cases use the wrapper function (vectorized_tt_dot).
    """
    num_objects, num_features = X.shape
    current_vectors = np.zeros((num_objects, ranks[1]))
    for obj_idx in range(num_objects):
        idx = 0
        for alpha_2 in range(ranks[1]):
            current_vectors[obj_idx, alpha_2] = linear_core_w[idx] + X[obj_idx, 0] * linear_core_w[idx + 1]
            idx += 2
    for dim in range(1, num_dims-1):
        prev_idx = idx
        next_vectors = np.zeros((num_objects, ranks[dim+1]))
        for obj_idx in range(num_objects):
            idx = prev_idx
            for alpha_2 in range(ranks[dim+1]):
                val = 0
                for alpha_1 in range(ranks[dim]):
                    curr_core = linear_core_w[idx] + X[obj_idx, dim] * linear_core_w[idx + 1]
                    val += current_vectors[obj_idx, alpha_1] * curr_core
                    idx += 2
                next_vectors[obj_idx, alpha_2] = val
        current_vectors = next_vectors
    dim = num_dims-1
    prev_idx = idx
    for obj_idx in range(num_objects):
        idx = prev_idx
        val = 0
        for alpha_1 in range(ranks[dim]):
            curr_core = linear_core_w[idx] + X[obj_idx, dim] * linear_core_w[idx + 1]
            idx += 2
            val += current_vectors[obj_idx, alpha_1] * curr_core
        result[obj_idx] = val


def vectorized_tt_dot(w, x):
    """Compute a dot products between a tensor w and subset tensors built from x.

    Returns a vector with the following number in the i-th element:
    tt.dot(w, subset_tensor(x[i, :]))
    """
    linear_core = _prepare_linear_core(w)
    res = np.zeros(x.shape[0])
    # On 64 bit Mac w.r is int32 for some reason and
    # jit version of _vectorized_tt_dot_ returns an error saying
    # that it got arguments of different datatype (int32 and int64).
    rank = w.r.astype(int)
    _vectorized_tt_dot_jit(linear_core, x, res, w.d, rank)
    return res
