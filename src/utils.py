from sklearn.metrics import roc_auc_score

# TODO: Change the order of inputs to all the objective functions in the library
# to (y_true, y_score) as in sklearn and get rid of this function.
def roc_auc_score_reversed(y_score, y_true):
    """All our losses assume (y_score, y_true) order, while sklearn
    assumes the reversed one.
    """
    return roc_auc_score(y_true, y_score)