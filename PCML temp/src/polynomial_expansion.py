# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np
import itertools


def polynomial_expansion(X, degree):
    """
    Computes the polynomial expansion of X up to the provided
    degree.

    Parameters
    ==========
    X: np.ndarray
        N x D matrix describing features.

    degree: int
        Maximal degree of the expansion.

    Returns
    =======
    X_poly: np.ndarray
        N x (D + Z) matrix describing features with polynomial expansion
        where Z is defined as the number of combinations with repetition
        of D indices.
    """

    if degree == 1:
        return X

    N, D = X.shape

    index_combinations = list(itertools.combinations_with_replacement(np.arange(D, dtype=int), degree))
    num_combinations = len(index_combinations)

    # X combined features only (does not contain "basic" features)
    # [x_1**2, x_1*x_2, x_2**2, ...]
    X_combinations = np.zeros((N, num_combinations))

    for (idx, indices) in enumerate(index_combinations):
        # extract columns for chosen parameters and multiply together
        X_combinations[:, idx] = np.prod(X[:, indices], axis=1)

    # X with basic and combined features
    # [x_1, x_2, ..., x_D, x_1**2, x_1*x_2, x_2**2, ...]
    X_poly = np.hstack((X, X_combinations))

    return X_poly
