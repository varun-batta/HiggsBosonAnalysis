# -*- coding: utf-8 -*-
"""
Gradient descent algorithms
"""

import numpy as np
import gradient
import cost


def compute_minibatch_indices(num_iters, num_samples, batch_size):
    """
    Returns minibatches of size batch_size. Enough minibatches are returned
    for max_iters iterations.

    Parameters
    ==========
    num_iters: int
        Number of iterations to perform.

    num_samples: int
        Number of samples in the dataset.

    batch_size: int
        Number of samples to use in each minibatch.

    Returns
    =======
    minibatch_indices_list: List[np.ndarray]
        List of num_iters minibatch indices.
    """

    # cannot takes batches that are larger than dataset
    assert(batch_size <= num_samples)

    num_epochs = int(np.ceil((num_iters * batch_size) / num_samples))

    # randomly shuffle num_epochs_needed FULL-batch indices
    batch_indices_list = [np.random.permutation(num_samples)
                          for i in range(num_epochs)]

    # concatenate all shuffled batches together
    batch_indices = np.hstack(batch_indices_list)

    # split shuffled batch indices into batch_size pieces
    # (last piece may be < batch_size)
    minibatch_indices_list = [batch_indices[i*batch_size : (i+1)*batch_size]
                              for i in range(num_iters)]

    return minibatch_indices_list


def gradient_descent(y, X, initial_w, max_iters, gamma, lamb, gradient_function, loss_function):
    """
    Generic gradient descent algorithm.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    gradient_function: Function[y, X, w, lamb]
        Function that computes the gradient.

    loss_function: Function[y, X, w, lamb]
        Function that computes the loss.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent.

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []

    for n_iter in range(max_iters):
        # current model parameters
        w = ws[n_iter]

        # compute current loss
        loss = loss_function(y, X, w, lamb)

        # compute gradient
        grad = gradient_function(y, X, w, lamb)

        # update model parameters by gradient
        w = w - gamma * grad

        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)

        if n_iter % 100 == 0:
            print("Gradient Descent({bi}/{ti}): loss = {l}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss))

    return losses, ws


def stochastic_gradient_descent(y, X, initial_w, batch_size, max_iters, gamma, lamb, gradient_function, loss_function):
    """
    Generic stochastic gradient descent algorithm.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    batch_size: int
        Number of samples to use for the gradient computation.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    gradient_function: Function[y, X, w, lamb]
        Function that computes the gradient.

    loss_function: Function[y, X, w, lamb]
        Function that computes the loss.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []

    num_samples = len(y)
    minibatch_indices_list = compute_minibatch_indices(max_iters, num_samples, batch_size)

    for (n_iter, minibatch_indices) in enumerate(minibatch_indices_list):
        # current model parameters
        w = ws[n_iter]

        # compute current loss
        loss = loss_function(y, X, w, lamb)

        # extract minibatch
        minibatch_y = y[minibatch_indices]
        minibatch_X = X[minibatch_indices]

        # compute gradient
        grad = gradient_function(minibatch_y, minibatch_X, w, lamb)

        # update model parameters by gradient
        w = w - gamma * grad

        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)

        if n_iter % 100 == 0:
            print("Stochastic Gradient Descent({bi}/{ti}): loss = {l}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss))

    return losses, ws


def mse_L0_gradient_descent(y, X, initial_w, max_iters, gamma, lamb=0):
    """
    MSE gradient descent subject to an L0 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    losses, ws = gradient_descent(y=y,
                                  X=X,
                                  initial_w=initial_w,
                                  max_iters=max_iters,
                                  gamma=gamma,
                                  lamb=lamb,
                                  gradient_function=gradient.mse_L0_gradient,
                                  loss_function=cost.mse_L0_loss)
    return losses, ws


def mse_L1_gradient_descent(y, X, initial_w, max_iters, gamma, lamb):
    """
    MSE gradient descent subject to an L1 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    losses, ws = gradient_descent(y=y,
                                  X=X,
                                  initial_w=initial_w,
                                  max_iters=max_iters,
                                  gamma=gamma,
                                  lamb=lamb,
                                  gradient_function=gradient.mse_L1_gradient,
                                  loss_function=cost.mse_L1_loss)
    return losses, ws


def mse_L2_gradient_descent(y, X, initial_w, max_iters, gamma, lamb):
    """
    MSE gradient descent subject to an L2 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    losses, ws = gradient_descent(y=y,
                                  X=X,
                                  initial_w=initial_w,
                                  max_iters=max_iters,
                                  gamma=gamma,
                                  lamb=lamb,
                                  gradient_function=gradient.mse_L2_gradient,
                                  loss_function=cost.mse_L2_loss)
    return losses, ws


def mae_L0_gradient_descent(y, X, initial_w, max_iters, gamma, lamb=0):
    """
    MAE gradient descent subject to an L0 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    losses, ws = gradient_descent(y=y,
                                  X=X,
                                  initial_w=initial_w,
                                  max_iters=max_iters,
                                  gamma=gamma,
                                  lamb=lamb,
                                  gradient_function=gradient.mae_L0_gradient,
                                  loss_function=cost.mae_L0_loss)
    return losses, ws


def mae_L1_gradient_descent(y, X, initial_w, max_iters, gamma, lamb):
    """
    MAE gradient descent subject to an L1 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    losses, ws = gradient_descent(y=y,
                                  X=X,
                                  initial_w=initial_w,
                                  max_iters=max_iters,
                                  gamma=gamma,
                                  lamb=lamb,
                                  gradient_function=gradient.mae_L1_gradient,
                                  loss_function=cost.mae_L1_loss)
    return losses, ws


def mae_L2_gradient_descent(y, X, initial_w, max_iters, gamma, lamb):
    """
    MAE gradient descent subject to an L2 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    losses, ws = gradient_descent(y=y,
                                  X=X,
                                  initial_w=initial_w,
                                  max_iters=max_iters,
                                  gamma=gamma,
                                  lamb=lamb,
                                  gradient_function=gradient.mae_L2_gradient,
                                  loss_function=cost.mae_L2_loss)
    return losses, ws


def logistic_L0_gradient_descent(y, X, initial_w, max_iters, gamma, lamb=0):
    """
    Logistic gradient descent subject to an L0 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    losses, ws = gradient_descent(y=y,
                                  X=X,
                                  initial_w=initial_w,
                                  max_iters=max_iters,
                                  gamma=gamma,
                                  lamb=lamb,
                                  gradient_function=gradient.logistic_L0_gradient,
                                  loss_function=cost.logistic_L0_loss)
    return losses, ws


def logistic_L1_gradient_descent(y, X, initial_w, max_iters, gamma, lamb):
    """
    Logistic gradient descent subject to an L1 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    losses, ws = gradient_descent(y=y,
                                  X=X,
                                  initial_w=initial_w,
                                  max_iters=max_iters,
                                  gamma=gamma,
                                  lamb=lamb,
                                  gradient_function=gradient.logistic_L1_gradient,
                                  loss_function=cost.logistic_L1_loss)
    return losses, ws


def logistic_L2_gradient_descent(y, X, initial_w, max_iters, gamma, lamb):
    """
    Logistic gradient descent subject to an L2 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    losses, ws = gradient_descent(y=y,
                                  X=X,
                                  initial_w=initial_w,
                                  max_iters=max_iters,
                                  gamma=gamma,
                                  lamb=lamb,
                                  gradient_function=gradient.logistic_L2_gradient,
                                  loss_function=cost.logistic_L2_loss)
    return losses, ws


def mse_L0_stochastic_gradient_descent(y, X, initial_w, batch_size, max_iters, gamma, lamb=0):
    """
    MSE stochastic gradient descent algorithm subject to an L0 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    batch_size: int
        Number of samples to use for the gradient computation.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    losses, ws = stochastic_gradient_descent(y=y,
                                             X=X,
                                             initial_w=initial_w,
                                             batch_size=batch_size,
                                             max_iters=max_iters,
                                             gamma=gamma,
                                             lamb=lamb,
                                             gradient_function=gradient.mse_L0_gradient,
                                             loss_function=cost.mse_L0_loss)
    return losses, ws


def mse_L1_stochastic_gradient_descent(y, X, initial_w, batch_size, max_iters, gamma, lamb):
    """
    MSE stochastic gradient descent algorithm subject to an L1 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    batch_size: int
        Number of samples to use for the gradient computation.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    losses, ws = stochastic_gradient_descent(y=y,
                                             X=X,
                                             initial_w=initial_w,
                                             batch_size=batch_size,
                                             max_iters=max_iters,
                                             gamma=gamma,
                                             lamb=lamb,
                                             gradient_function=gradient.mse_L1_gradient,
                                             loss_function=cost.mse_L1_loss)
    return losses, ws


def mse_L2_stochastic_gradient_descent(y, X, initial_w, batch_size, max_iters, gamma, lamb):
    """
    MSE stochastic gradient descent algorithm subject to an L2 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    batch_size: int
        Number of samples to use for the gradient computation.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    losses, ws = stochastic_gradient_descent(y=y,
                                             X=X,
                                             initial_w=initial_w,
                                             batch_size=batch_size,
                                             max_iters=max_iters,
                                             gamma=gamma,
                                             lamb=lamb,
                                             gradient_function=gradient.mse_L2_gradient,
                                             loss_function=cost.mse_L2_loss)
    return losses, ws


def mae_L0_stochastic_gradient_descent(y, X, initial_w, batch_size, max_iters, gamma, lamb=0):
    """
    MAE stochastic gradient descent algorithm subject to an L0 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    batch_size: int
        Number of samples to use for the gradient computation.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    losses, ws = stochastic_gradient_descent(y=y,
                                             X=X,
                                             initial_w=initial_w,
                                             batch_size=batch_size,
                                             max_iters=max_iters,
                                             gamma=gamma,
                                             lamb=lamb,
                                             gradient_function=gradient.mae_L0_gradient,
                                             loss_function=cost.mae_L0_loss)
    return losses, ws


def mae_L1_stochastic_gradient_descent(y, X, initial_w, batch_size, max_iters, gamma, lamb):
    """
    MAE stochastic gradient descent algorithm subject to an L1 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    batch_size: int
        Number of samples to use for the gradient computation.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    losses, ws = stochastic_gradient_descent(y=y,
                                             X=X,
                                             initial_w=initial_w,
                                             batch_size=batch_size,
                                             max_iters=max_iters,
                                             gamma=gamma,
                                             lamb=lamb,
                                             gradient_function=gradient.mae_L1_gradient,
                                             loss_function=cost.mae_L1_loss)
    return losses, ws


def mae_L2_stochastic_gradient_descent(y, X, initial_w, batch_size, max_iters, gamma, lamb):
    """
    MAE stochastic gradient descent algorithm subject to an L2 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    batch_size: int
        Number of samples to use for the gradient computation.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    losses, ws = stochastic_gradient_descent(y=y,
                                             X=X,
                                             initial_w=initial_w,
                                             batch_size=batch_size,
                                             max_iters=max_iters,
                                             gamma=gamma,
                                             lamb=lamb,
                                             gradient_function=gradient.mae_L2_gradient,
                                             loss_function=cost.mae_L2_loss)


def logistic_L0_stochastic_gradient_descent(y, X, initial_w, batch_size, max_iters, gamma, lamb):
    """
    Logistic stochastic gradient descent subject to an L0 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    batch_size: int
        Number of samples to use for the gradient computation.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    losses, ws = stochastic_gradient_descent(y=y,
                                             X=X,
                                             initial_w=initial_w,
                                             batch_size=batch_size,
                                             max_iters=max_iters,
                                             gamma=gamma,
                                             lamb=lamb,
                                             gradient_function=gradient.logistic_L0_gradient,
                                             loss_function=cost.logistic_L0_loss)
    return losses, ws


def logistic_L1_stochastic_gradient_descent(y, X, initial_w, batch_size, max_iters, gamma, lamb):
    """
    Logistic stochastic gradient descent subject to an L1 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    batch_size: int
        Number of samples to use for the gradient computation.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    losses, ws = stochastic_gradient_descent(y=y,
                                             X=X,
                                             initial_w=initial_w,
                                             batch_size=batch_size,
                                             max_iters=max_iters,
                                             gamma=gamma,
                                             lamb=lamb,
                                             gradient_function=gradient.logistic_L1_gradient,
                                             loss_function=cost.logistic_L1_loss)
    return losses, ws


def logistic_L2_stochastic_gradient_descent(y, X, initial_w, batch_size, max_iters, gamma, lamb):
    """
    Logistic stochastic gradient descent subject to an L2 regularizer.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    initial_w: np.ndarray
        D-vector describing model parameters.

    batch_size: int
        Number of samples to use for the gradient computation.

    max_iters: int
        Maximum number of iterations to perform.

    gamma: np.float64
        Learning rate.

    lamb: np.float64
        Regularization factor.

    Returns
    =======
    losses: List[np.float64]
        List of losses obtained through each iteration
        of the gradient descent,

    ws: List[np.ndarray]
        List of model parameters obtained through each iteration
        of the gradient descent.
    """

    losses, ws = stochastic_gradient_descent(y=y,
                                             X=X,
                                             initial_w=initial_w,
                                             batch_size=batch_size,
                                             max_iters=max_iters,
                                             gamma=gamma,
                                             lamb=lamb,
                                             gradient_function=gradient.logistic_L2_gradient,
                                             loss_function=cost.logistic_L2_loss)
    return losses, ws
