# -*- coding: utf-8 -*-
"""
Gradient computations
"""

import numpy as np
import logistic


def gradient_L1_regularizer(w, lamb):
    """
    Computes an L1 regularizer.

    Parameters
    ==========
    w: np.ndarray
        D-vector describing model parameters.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    res: np.float64
        L1 regularizer
    """

    w_reg = w[1:]
    D = w_reg.shape[0]
    regularizer = (lamb / D) * np.hstack((0, np.sign(w_reg)))

    return regularizer


def gradient_L2_regularizer(w, lamb):
    """
    Computes an L2 regularizer.

    Parameters
    ==========
    w: np.ndarray
        D-vector describing model parameters.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    res: np.float64
        L1 regularizer
    """

    w_reg = w[1:]
    D = w_reg.shape[0]
    regularizer = (2 / D) * lamb * np.hstack((0, w_reg))

    return regularizer


def mse_L0_gradient(y, X, w, lamb=0):
    """
    Computes the gradient of the MSE loss function subject to an L0
    regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    w: np.ndarray
        D-vector describing model parameters.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    res: np.ndarray
        D-vector of gradient (1 dimension for each model parameter)
    """

    N, D = X.shape
    e = y - X.dot(w)
    grad = -(1 / N) * X.T.dot(e)

    return grad


def mse_L1_gradient(y, X, w, lamb):
    """
    Computes the gradient of the MSE loss function subject to an L1
    regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    w: np.ndarray
        D-vector describing model parameters.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    res: np.ndarray
        D-vector of gradient (1 dimension for each model parameter)
    """

    grad = mse_L0_gradient(y, X, w) + gradient_L1_regularizer(w, lamb)

    return grad


def mse_L2_gradient(y, X, w, lamb):
    """
    Computes the gradient of the MSE loss function subject to an L2
    regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    w: np.ndarray
        D-vector describing model parameters.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    res: np.ndarray
        D-vector of gradient (1 dimension for each model parameter)
    """

    grad = mse_L0_gradient(y, X, w) + gradient_L2_regularizer(w, lamb)

    return grad


def mae_L0_gradient(y, X, w, lamb=0):
    """
    Computes the gradient of the MAE loss function subject to an L0 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    w: np.ndarray
        D-vector describing model parameters.

    Returns
    =======
    res: np.ndarray
        D-vector of gradient (1 dimension for each model parameter)
    """

    N, D = X.shape
    e = y - X.dot(w)
    grad = -(1 / N) * X.T.dot(np.sign(e))

    return grad


def mae_L1_gradient(y, X, w, lamb):
    """
    Computes the gradient of the MAE loss function subject to an L1
    regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    w: np.ndarray
        D-vector describing model parameters.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    res: np.ndarray
        D-vector of gradient (1 dimension for each model parameter)
    """

    grad = mae_L0_gradient(y, X, w) + gradient_L1_regularizer(w, lamb)

    return grad


def mae_L2_gradient(y, X, w, lamb):
    """
    Computes the gradient of the MAE loss function subject to an L2
    regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    w: np.ndarray
        D-vector describing model parameters.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    res: np.ndarray
        D-vector of gradient (1 dimension for each model parameter)
    """

    grad = mae_L0_gradient(y, X, w) + gradient_L2_regularizer(w, lamb)

    return grad


def logistic_L0_gradient(y, X, w, lamb=0):
    """
    Computes the gradient for regularized logistic regression subject to an L0
    regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    w: np.ndarray
        D-vector describing model parameters.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    res: np.ndarray
        D-vector of gradient (1 dimension for each model parameter)
    """

    N, D = X.shape
    grad = (1 / N) * X.T.dot(logistic.sigmoid(X, w) - y)

    return grad


def logistic_L1_gradient(y, X, w, lamb):
    """
    Computes the gradient for regularized logistic regression subject to an L1
    regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    w: np.ndarray
        D-vector describing model parameters.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    res: np.ndarray
        D-vector of gradient (1 dimension for each model parameter)
    """

    grad = logistic_L0_gradient(y, X, w) + gradient_L1_regularizer(w, lamb)

    return grad


def logistic_L2_gradient(y, X, w, lamb):
    """
    Computes the gradient for regularized logistic regression subject to an L2
    regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    w: np.ndarray
        D-vector describing model parameters.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    res: np.ndarray
        D-vector of gradient (1 dimension for each model parameter)
    """

    grad = logistic_L0_gradient(y, X, w) + gradient_L2_regularizer(w, lamb)

    return grad
