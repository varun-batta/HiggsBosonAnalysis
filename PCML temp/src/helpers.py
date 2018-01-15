# -*- coding: utf-8 -*-
"""
Some helper functions.
"""

import numpy as np
import datetime

import logistic


def standardize(X, mean_X=None, std_X=None):
    """
    Standardize the data set.

    Parameters
    ==========
    X: np.ndarray
        N x D-matrix describing features.

    mean_X: np.ndarray
        D-vector of means of the N samples.

    std_X: np.ndarray
        D-vector of standard deviations of the N samples.


    Returns
    =======
    X: np.ndarray
        Standardized N x D-matrix describing features.

    mean_X: np.ndarray
        D-vector of means of the N samples.

    std_X: np.ndarray
        D-vector of standard deviations of the N samples.
    """

    if mean_X is None:
        mean_X = np.mean(X, axis=0)

    # center around mean
    X = X - mean_X

    if std_X is None:
        std_X = np.std(X, axis=0)

    # normalize by standard deviation
    X[:, std_X>0] = X[:, std_X>0] / std_X[std_X>0]

    return X, mean_X, std_X


def add_offset_parameter(X):
    """
    Adds the offset parameter to the input matrix.

    Parameters
    ==========
    X: np.ndarray
        N x D-matrix describing features.

    Returns
    =======
    X: np.ndarray
        N x (D+1)-matrix describing features with additional
        offset parameter (leading column of 1s)
    """

    N, D = X.shape
    offset_column = np.ones((N, 1))

    return np.hstack((offset_column, X))


def exec_time(func, *args):
    """
    Executes and computes the total runtime of the input function with the
    supplied arguments.

    Parameters
    ----------
    func: Function
        Function to execute

    args: Arguments
        Function arguments

    Returns
    -------
    (func_res, func_name, total_time): (Function output, String, Float)
        Returns a tuple containing the output of the function, the function's
        name, and the total execution time of the function.
    """

    start_time = datetime.datetime.now()
    func_res = func(*args)
    end_time = datetime.datetime.now()

    func_name = func.__name__
    total_time = (end_time - start_time).total_seconds()

    return (func_res, func_name, total_time)


def prediction_accuracy(y, X, w):
    """
    Returns a K-fold cross validation generator. Each item obtained from the
    generator consists of (train_y, train_X, test_y, test_X) tuples.

    The proportion of test     samples is N // fold_count.
    The proportion of training samples is N - (N // fold_count).

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
    correct_count: int
        Number of correct predictions.

    total_count: int
        Total number of samples.
    """

    N, D = X.shape

    y_probability = logistic.sigmoid(X, w)

    category_0_condition = y_probability < 0.5
    category_1_condition = np.logical_not(category_0_condition)

    y_predict = np.zeros(N)
    y_predict[category_0_condition] = 0
    y_predict[category_1_condition] = 1

    correct_count = np.sum(y_predict == y)
    total_count = y.shape[0]

    return (correct_count, total_count)


def split_data(X, y, ratio, seed=None):
    '''
    Returns both test and train data splitted into two subsets according to ratio.

    Parameters
    ==========
    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    ratio: float
        ratio to split the data. ratio should be in range [0.0, 1.0]

    seed: int
        rng seed

    Returns
    =======
    x_tr: np.ndarray
        (N * ratio) x D matrix with training data

    x_te: np.ndarray
        (N * (1 - ratio)) x D matrix with training data

    y_tr: np.ndarray
        (N * ratio)-vector of training labels

    y_te: np.ndarray
        (N * (1 - ratio))-vector of test labels
    '''

    # set seed
    np.random.seed(seed)

    ntr = round(y.shape[0] * ratio)
    ind = np.random.permutation(range(y.shape[0]))

    x_tr = X[ind[:ntr]]
    x_te = X[ind[ntr:]]
    y_tr = y[ind[:ntr]]
    y_te = y[ind[ntr:]]

    return (x_tr, x_te, y_tr, y_te)
