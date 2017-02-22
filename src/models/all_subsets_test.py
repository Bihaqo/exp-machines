import random
import unittest
import numpy as np
import copy
import tt
from tt.riemannian import riemannian
import all_subsets


class AllSubsetsTest(unittest.TestCase):

    def setUp(self):
        # Reproducibility.
        random.seed(0)
        np.random.seed(2)

    def test_project_all_subsets(self):
        reg = 0.7
        for debug_mode in [False, True]:
            w = tt.rand([2, 2, 2, 2], 4, [1, 2, 3, 2, 1])
            X = np.random.randn(10, 4)
            exact_answ = reg * w.full()
            for obj_idx in range(10):
                obj = all_subsets.subset_tensor(X[obj_idx, :])
                exact_answ += riemannian.project(w, obj).full()

            res = all_subsets.project_all_subsets(w, X, reg=reg, debug=debug_mode)
            np.testing.assert_array_almost_equal(res.full(), exact_answ)

    def test_tensorize_linear_init(self):
        coef = [0, 3, 0.2, 0, -12]
        intercept = 5
        w_coef = all_subsets.tensorize_linear_init(coef, intercept)
        desired = np.zeros(w_coef.n)
        desired[0, 0, 0, 0, 0] = intercept
        desired[1, 0, 0, 0, 0] = coef[0]
        desired[0, 1, 0, 0, 0] = coef[1]
        desired[0, 0, 1, 0, 0] = coef[2]
        desired[0, 0, 0, 1, 0] = coef[3]
        desired[0, 0, 0, 0, 1] = coef[4]
        np.testing.assert_almost_equal(desired, w_coef.full())

    def test_categorical_tensorize_linear_init(self):
        coef = [[0, 1], [2, 3, 4], [-12, 0.4]]
        intercept = 5
        w_coef = all_subsets.categorical_tensorize_linear_init(coef, intercept)
        desired = np.zeros(w_coef.n)
        desired[0, 0, 0] = intercept
        desired[1, 0, 0] = coef[0][0]
        desired[2, 0, 0] = coef[0][1]
        desired[0, 1, 0] = coef[1][0]
        desired[0, 2, 0] = coef[1][1]
        desired[0, 3, 0] = coef[1][2]
        desired[0, 0, 1] = coef[2][0]
        desired[0, 0, 2] = coef[2][1]
        np.testing.assert_almost_equal(desired, w_coef.full())

    def test_gradient_wrt_core(self):
        w = tt.rand([2, 2, 2], 3, [1, 2, 2, 1])
        x_1 = all_subsets.subset_tensor([3, 4, 5])
        x_2 = all_subsets.subset_tensor([-1, 12, 5])
        X = np.array([[3, 4, 5], [-1, 12, 5]])
        eps = 1e-8

        def loss(core):
            new_w = w.copy()
            new_w.core = copy.copy(core)
            res = (tt.dot(new_w, x_1))**2 # Quadratic.
            res += tt.dot(new_w, x_2)     # Linear.
            return res

        # Derivatives of the quadratic and linear functions in the loss.
        dfdz = [2 * tt.dot(w, x_1), 1]

        core = w.core
        value = loss(core)
        numerical_grad = np.zeros(len(core))
        for i in range(len(core)):
            new_core = copy.copy(core)
            new_core[i] += eps
            numerical_grad[i] = (loss(new_core) - value) / eps

        w_cores = tt.tensor.to_list(w)
        gradient = all_subsets.gradient_wrt_cores(w_cores, X, dfdz)
        np.testing.assert_array_almost_equal(numerical_grad, gradient, decimal=3)

    def test_vectorized_tt_dot(self):
        w = tt.rand([2, 2, 2, 2], 4, [1, 2, 3, 2, 1])
        X = np.random.randn(10, 4)
        exact_answ = np.zeros(10)
        for obj_idx in range(10):
            obj = all_subsets.subset_tensor(X[obj_idx, :])
            exact_answ[obj_idx] = tt.dot(w, obj)
        res = all_subsets.vectorized_tt_dot(w, X)
        np.testing.assert_array_almost_equal(res, exact_answ)

    def test_categorical_vectorized_tt_dot(self):
        w = tt.rand([3, 4, 5, 6], 4, [1, 2, 4, 5, 1])
        X_1 = np.random.randint(2, size=(10, 1))
        X_2 = np.random.randint(3, size=(10, 1))
        X_3 = np.random.randint(4, size=(10, 1))
        X_4 = np.random.randint(5, size=(10, 1))
        X = np.hstack((X_1, X_2, X_3, X_4)) + 1
        exact_answ = np.zeros(10)
        for obj_idx in range(10):
            obj = all_subsets.categorical_subset_tensor(X[obj_idx, :], [2, 3, 4, 5])
            exact_answ[obj_idx] = tt.dot(w, obj)
        res = all_subsets.categorical_vectorized_tt_dot(w, X)
        np.testing.assert_array_almost_equal(res, exact_answ)


if __name__ == '__main__':
    unittest.main()
