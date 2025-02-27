# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    poly=np.zeros((len(x),degree+1))
    for i in range (len(x)):
        for j in range(degree+1):
            poly[i,j]=x[i]**j
    return poly