This is an implementation of Exponential Machines (the paper is coming soon).


# Prerequisites
* The Python version of the [TT-toolbox](https://github.com/oseledets/ttpy).
* [Scikit-learn](http://scikit-learn.org/stable/)
* [Numba](http://numba.pydata.org/)

# Usage
The interface is the same as of Scikit-learn models. To train a model with logistic loss, TT-rank equal 4  using the Riemannian solver for 10 iteration use the following code:
```
model = TTRegression('all-subsets', 'logistic', rank=4, solver='riemannian-sgd', max_iter=10, verbose=2)
model.fit(X_train, y_train)
```

# Experiments from the paper
The code to reproduce the experiments is in the  `experiments` folder, one Jupyter Notebook per each experiment.
