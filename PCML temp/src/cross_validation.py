# -*- coding: utf-8 -*-
"""
Cross validation algorithms
"""

import numpy as np

import helpers


def cross_validate_generator(id, y, X, fold_count, seed):
    """
    Returns a K-fold cross validation generator. Each item obtained from the
    generator consists of (train_y, train_X, test_y, test_X) tuples.

    The proportion of test     samples is N // fold_count.
    The proportion of training samples is N - (N // fold_count).

    Parameters
    ==========
    id: np.ndarray
        N-vector of sample identifiers.

    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    fold_count: int
        Ratio between full data set and test set.

    seed: int
        Seed for random number generator.

    Returns
    =======
    (id_train, y_train, X_train, id_test, y_test, X_test): Generator[(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)]
        Generator of training and testing labels and samples.
    """

    N, D = X.shape
    fold_size = N // fold_count

    # randomly shuffle data
    np.random.seed(seed)
    random_shuffle = np.random.permutation(N)
    id_shuffle = id[random_shuffle]
    y_shuffle = y[random_shuffle]
    X_shuffle = X[random_shuffle]

    # used for indexing later
    full_range = np.arange(N, dtype=int)

    for fold_idx in range(fold_count):
        # test indices (will be larger than N, must wrap later)
        test_range_lower = fold_idx * fold_size
        test_range_upper = ((fold_idx + 1) * fold_size)
        test_range = np.arange(test_range_lower, test_range_upper, dtype=int)

        # train indices (will be larger than N, must wrap later)
        train_range_lower = (fold_idx + 1) * fold_size
        train_range_upper = ((fold_idx + fold_count) * fold_size)
        train_range = np.arange(train_range_lower, train_range_upper, dtype=int)

        # indices (smaller than N)
        test_indices = np.take(full_range, test_range, mode="wrap")
        train_indices = np.take(full_range, train_range, mode="wrap")

        # extract test and train data
        id_test = id_shuffle[test_indices]
        y_test = y_shuffle[test_indices]
        X_test = X_shuffle[test_indices]
        id_train = id_shuffle[train_indices]
        y_train = y_shuffle[train_indices]
        X_train = X_shuffle[train_indices]

        # If there is only 1 fold, we want to have all the data as part of the
        # training set, but we allocate data first to the test set. In this case
        # we just swap what is considered test to become train. This way train data
        # has all the data samples, and test is empty.
        if fold_count == 1:
            yield (id_test, y_test, X_test, id_train, y_train, X_train)
        else:
            yield (id_train, y_train, X_train, id_test, y_test, X_test)


def cross_validate(id, y, X, fold_count, seed, gd_func, initial_w, max_iters, gamma, lamb):
    """
    Computes test and train predictions after K-fold cross-validation.
    The algorithm used to perform the optimization is gradient descent.

    Parameters
    ==========
    id: np.ndarray
        N-vector of sample identifiers.

    y: np.ndarray
        N-vector describing prediction.

    X: np.ndarray
        N x D matrix describing features.

    fold_count: int
        Ratio between full data set and test set.

    seed: int
        Seed for random number generator.

    gd_func: Function [y, X, initial_w, max_iters, gamma, lamb]
        Gradient descent function.

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
    w_stars: np.ndarray
        fold_count x D matrix of best model parameters found at each fold.

    train_correct_ratios: np.ndarray
        fold_count-vector of training accuracy at each fold.

    test_correct_ratios: np.ndarray
        fold_count-vector of test accuracy at each fold.
    """

    N, D = X.shape

    w_stars = np.zeros((fold_count, D))
    train_correct_ratios = np.zeros(fold_count)
    test_correct_ratios = np.zeros(fold_count)

    for (fold_idx, (id_train, y_train, X_train, id_test, y_test, X_test)) in \
    enumerate(cross_validate_generator(id, y, X, fold_count, seed)):

        # learn model with training set
        (losses, ws) = gd_func(y_train, X_train, initial_w, max_iters, gamma, lamb)

        # "optimal" model parameters found
        w_star = ws[-1]

        # training set accuracy
        (train_correct_count, train_total_count) = helpers.prediction_accuracy(y_train, X_train, w_star)
        train_correct_ratio = train_correct_count / train_total_count

        # test set accuracy
        (test_correct_count, test_total_count) = helpers.prediction_accuracy(y_test, X_test, w_star)
        test_correct_ratio = test_correct_count / test_total_count

        # keep track of computed values
        w_stars[fold_idx] = w_star
        train_correct_ratios[fold_idx] = train_correct_ratio
        test_correct_ratios[fold_idx] = test_correct_ratio

    return (w_stars, train_correct_ratios, test_correct_ratios)
