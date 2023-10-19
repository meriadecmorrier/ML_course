# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e=y-np.dot(tx,w)
    return (1/(2*len(y)))*np.dot(e,e.T)

def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    
    w=np.linalg.solve(np.dot(tx.T,tx),np.dot(tx.T,y))
    loss=compute_mse(y,tx,w)
    return w,loss
