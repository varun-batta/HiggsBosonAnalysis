# -*- coding: utf-8 -*-
"""
Regression functions.
"""

import numpy as np


def least_squares(y, X):
    """
    Calculate the least squares solution of Xw = y using the normal equations.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    Returns
    =======
    w: np.ndarray
        D-vector describing model parameters under least squares.
    """

    # # non-numerically stable version
    # w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    # numerically stable version
    A = X.T.dot(X) # D x D
    b = X.T.dot(y) # D X 1
    w = np.linalg.solve(A, b)

    return w


def ridge_regression(y, X, lamb):
    """
    Ridge regression algorithm.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    lamb: np.float64
        Regularization parameter.

    Returns
    =======
    w: np.ndarray
        D-vector describing model parameters under least squares.
    """

    (N, M) = X.shape
    lamb_prime = 2 * N * lamb

    # # non-numerically stable version
    # w = np.linalg.inv(X.T.dot(X) + lamb_prime * np.eye(M)).dot(X.T).dot(y)

    # numerically stable version
    A = X.T.dot(X) + lamb_prime * np.diag(np.hstack((0, np.ones(M - 1))))
    b = X.T.dot(y)
    w = np.linalg.solve(A, b)

    return w
