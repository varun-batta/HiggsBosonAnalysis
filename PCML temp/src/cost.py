# -*- coding: utf-8 -*-
"""
Loss functions
"""

import numpy as np


def cost_L1_regularizer(w, lamb):
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
    regularizer = (lamb / D) * np.sum(np.abs(w_reg))

    return regularizer


def cost_L2_regularizer(w, lamb):
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
        L2 regularizer
    """

    w_reg = w[1:]
    D = w_reg.shape[0]
    regularizer = (lamb / D) * w_reg.T.dot(w_reg)

    return regularizer


def mse_L0_loss(y, X, w, lamb=0):
    """
    Computes the Mean Squared Error (MSE) of the given linear
    model with respect to its parameters w subject to an L0 regularizer.

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
    res: np.float64
        Mean Squared Error.
    """

    N, D = X.shape
    e = y - X.dot(w)
    mse = (1 / (2 * N)) * e.T.dot(e)

    return mse


def mse_L1_loss(y, X, w, lamb):
    """
    Computes the Mean Squared Error (MSE) of the given linear
    model with respect to its parameters w subject to an L1 regularizer.

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
    res: np.float64
        Mean Squared Error.
    """

    mse = mse_L0_loss(y, X, w) + cost_L1_regularizer(w, lamb)

    return mse


def mse_L2_loss(y, X, w, lamb):
    """
    Computes the Mean Squared Error (MSE) of the given linear
    model with respect to its parameters w subject to an L2 regularizer.

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
    res: np.float64
        Mean Squared Error.
    """

    mse = mse_L0_loss(y, X, w) + cost_L2_regularizer(w, lamb)

    return mse


def mae_L0_loss(y, X, w, lamb=0):
    """
    Computes the Mean Absolute Error (MAE) of the give linear
    model with respect to its parameters w subject to an L0 regularizer.

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
    res: np.float64
        Mean Absolute Error.
    """

    N, D = X.shape
    e = y - X.dot(w)
    mae = (1 / N) * np.sum(np.fabs(e))

    return mae


def mae_L1_loss(y, X, w, lamb):
    """
    Computes the Mean Absolute Error (MAE) of the give linear
    model with respect to its parameters w subject to an L1 regularizer.

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
    res: np.float64
        Mean Absolute Error.
    """

    mae = mae_L0_loss(y, X, w) + cost_L1_regularizer(w, lamb)

    return mae


def mae_L2_loss(y, X, w, lamb):
    """
    Computes the Mean Absolute Error (MAE) of the give linear
    model with respect to its parameters w subject to an L2 regularizer.

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
    res: np.float64
        Mean Absolute Error.
    """

    mae = mae_L0_loss(y, X, w) + cost_L2_regularizer(w, lamb)

    return mae


def rmse_L0_loss(y, X, w, lamb=0):
    """
    Computes the Root Mean Squared Error (RMSE) of the given linear
    model with respect to its parameters w.

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
    res: np.float64
        Root Mean Squared Error.
    """

    return np.sqrt(2 * mse_L0_loss(y, X, w))


def log_1_plus_exp(s):
    """
    Helper function to compute log(1 + exp(s)) without overflows or underflows

    Reference: http://stackoverflow.com/questions/20085768/avoiding-numerical-overflow-when-calculating-the-value-and-gradient-of-the-logis

    Parameters
    ==========
    s: np.ndarray
        N-vector of exponents.

    Returns
    =======
    res: np.ndarray
        N-vector of results.
    """

    neg_condition = s < 0
    pos_condition = s >= 0

    res = np.zeros(s.shape[0])
    res[neg_condition] = np.log(1 + np.exp(s[neg_condition]))
    res[pos_condition] = s[pos_condition] + np.log(1 + np.exp(-s[pos_condition]))

    return res


def logistic_L0_loss(y, X, w, lamb=0):
    """
    Computes the logistic cost of the give linear
    model with respect to its parameters w subject to an L0 regularizer.

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
    res: np.float64
        Logistic cost.
    """

    N, D = X.shape
    Xw = X.dot(w)
    cost = (1 / N) * np.sum(log_1_plus_exp(Xw) - y * Xw)

    return cost


def logistic_L1_loss(y, X, w, lamb):
    """
    Computes the logistic cost of the give linear
    model with respect to its parameters w subject to an L1 regularizer.

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
    res: np.float64
        Logistic cost.
    """

    cost = logistic_L0_loss(y, X, w) + cost_L1_regularizer(w, lamb)

    return cost


def logistic_L2_loss(y, X, w, lamb):
    """
    Computes the logistic cost of the give linear
    model with respect to its parameters w subject to an L2 regularizer.

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
    res: np.float64
        Logistic cost.
    """

    cost = logistic_L0_loss(y, X, w) + cost_L2_regularizer(w, lamb)

    return cost
