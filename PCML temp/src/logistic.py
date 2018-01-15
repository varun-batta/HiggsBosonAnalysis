# -*- coding: utf-8 -*-
"""
Loss functions
"""

import numpy as np


def sigmoid(X, w):
    """
    Computes the sigmoid function of X.dot(w)

    Parameters
    ==========
    X: np.ndarray
        N x D matrix describing features.

    w: np.ndarray
        D-vector describing model parameters.

    Returns
    =======
    res: np.ndarray
        N-vector of sigmoid function applied to each prediction
        obtained by X.dot(w)
    """

    Xw = X.dot(w)
    sig = 1 / (1 + np.exp(-Xw))

    return sig
