# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """

    a=lambda_*np.identity(len(y))
    b=np.dot(tx.T,tx)
    left_term=np.dot(tx.T,tx)+2*len(y)*lambda_*np.identity(tx.shape[1])
    right_term=np.dot(tx.T,y)
    w=np.linalg.solve(left_term,right_term)
    return(w)