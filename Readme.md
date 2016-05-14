This is an implementation of Exponential Machines form the paper Tensor Train polynomial models via Riemannian optimization [\[1605.03795\]](https://arxiv.org/abs/1605.03795).

The main idea is to use a full exponentially-large polynomial model with all interactions of every order. To deal with exponential complexity we represent and learn the tensor of parameters in the Tensor Train (TT) format.


# Dependencies
* The Python version of the [TT-toolbox](https://github.com/oseledets/ttpy).
* [Scikit-learn](http://scikit-learn.org/stable/)
* [NumPy](http://www.numpy.org/)
* [Numba](http://numba.pydata.org/)
* [Accelerate](https://docs.continuum.io/accelerate/index) is not required, but makes everything faster.

# Usage
The interface is the same as of Scikit-learn models. To train a model with logistic loss, TT-rank equal 4  using the Riemannian solver for 10 iteration use the following code:
```
model = TTRegression('all-subsets', 'logistic', rank=4, solver='riemannian-sgd', max_iter=10, verbose=2)
model.fit(X_train, y_train)
```

# Experiments from the paper
The code to reproduce the experiments is in the  `experiments` folder, one Jupyter Notebook per each experiment.
