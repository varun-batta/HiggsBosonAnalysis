import numpy as np

import cost
import gradient_descent
import normal_equations


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    tx: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    Returns
    =======
    w: np.ndarray
        D-vector of best model parameters obtained through the gradient descent.

    loss: np.float64
        Best loss obtained through the gradient descent.
    """

    (losses, ws) = gradient_descent.mse_L0_gradient_descent(y=y,
                                                            X=tx,
                                                            initial_w=initial_w,
                                                            max_iters=max_iters,
                                                            gamma=gamma)
    w = ws[-1]
    loss = losses[-1]

    return (w, loss)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    tx: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    Returns
    =======
    w: np.ndarray
        D-vector of best model parameters obtained through the stochastic
        gradient descent.

    loss: np.float64
        Best loss obtained through the stochastic gradient descent.
    """

    (losses, ws) = gradient_descent.mse_L0_stochastic_gradient_descent(y=y,
                                                                       X=tx,
                                                                       initial_w=initial_w,
                                                                       batch_size=1,
                                                                       max_iters=max_iters,
                                                                       gamma=gamma)
    w = ws[-1]
    loss = losses[-1]

    return (w, loss)


def least_squares(y, tx):
    """
    Least squares regression using normal equations.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    tx: np.ndarray
        N x D matrix describing features.

    Returns
    =======
    w: np.ndarray
        D-vector of best model parameters obtained through normal equations.

    loss: np.float64
        Best loss obtained through normal equations.
    """

    w = normal_equations.least_squares(y=y, X=tx)
    loss = cost.mse_L0_loss(y=y, X=tx, w=w)

    return (w, loss)


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    tx: np.ndarray
        N x D matrix describing features.

    lambda_: np.float64
        Regularization parameter.

    Returns
    =======
    w: np.ndarray
        D-vector of best model parameters obtained through normal equations.

    loss: np.float64
        Best loss obtained through normal equations.
    """

    w = normal_equations.ridge_regression(y=y, X=tx, lamb=lambda_)
    loss = cost.mse_L2_loss(y=y, X=tx, w=w, lamb=lambda_)

    return (w, loss)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    tx: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    Returns
    =======
    w: np.ndarray
        D-vector of best model parameters obtained through the gradient descent.

    loss: np.float64
        Best loss obtained through the gradient descent.
    """

    (losses, ws) = gradient_descent.logistic_L0_gradient_descent(y=y,
                                                                 X=tx,
                                                                 initial_w=initial_w,
                                                                 max_iters=max_iters,
                                                                 gamma=gamma)
    w = ws[-1]
    loss = losses[-1]

    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    tx: np.ndarray
        N x D matrix describing features.

    lambda_: np.float64
        Regularization parameter.

    initial_w: np.ndarray
        D-vector describing model parameters.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    Returns
    =======
    w: np.ndarray
        D-vector of best model parameters obtained through the gradient descent.

    loss: np.float64
        Best loss obtained through the gradient descent.
    """

    (losses, ws) = gradient_descent.logistic_L2_gradient_descent(y=y,
                                                                 X=tx,
                                                                 initial_w=initial_w,
                                                                 max_iters=max_iters,
                                                                 gamma=gamma,
                                                                 lamb=lambda_)
    w = ws[-1]
    loss = losses[-1]

    return (w, loss)
