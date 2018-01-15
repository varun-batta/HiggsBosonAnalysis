# -*- coding: utf-8 -*-
"""
Main script.
"""

import numpy as np
import pickle

import clean_data
import cross_validation
import gradient_descent
import helpers
import polynomial_expansion
import proj1_helpers

################################################################################
#                                    constants                                 #
################################################################################
# data file
DATA_TRAIN_PATH = '../data/train_clean_avg.csv'
invalid_field_value = -999

# polynomial expansion
degree = 2

# cross-validation
fold_count = 1
seed = 2

# optimization
gd_func = gradient_descent.logistic_L2_gradient_descent
max_iters = 7000
gamma = 0.08

# lambdas (to find with grid search)
lambdas = np.linspace(10, 13, num=6)
lambda_best = 0

################################################################################
#                                    read data                                 #
################################################################################
(y, X, id) = proj1_helpers.load_csv_data(DATA_TRAIN_PATH, sub_sample=False)

# y is categorical, so we want integers (-1, 1) instead of floats (-1.0, 1.0)
# Modified here instead of in load_csv_data, because we don't know if we have the
# right to change the provided functions.
y = y.astype(int)

# The formulas used for the cost and gradients of the logistic function expect
# categories that are 0/1 for some terms to disappear in the equations.
y[np.where(y == -1)] = 0

################################################################################
#                                   clean data                                 #
################################################################################
# one-hot coding for "PRI_jet_num" (column 22)
(id, y, X) = clean_data.one_hot_PRI_jet_num(id, y, X)

# # keep only columns with highest discriminative power
# cleanInds = np.array([0, 1, 4, 10, 11, 12, 22, 24, 27]) # most discriminative power (visually)
# inds = np.array([0, 1, 4, 10, 11, 12, 23, 26, 29, 30, 31, 32])
# X = clean_data.keep_dimensions(X, inds)

# # use nearest neighbor (takes ~30 minutes for train data)
# (id, y, X) = clean_data.nearest_neighbour(id, y, X, invalid_field_value)

# (id, y, X) = clean_data.avg_incomplete_cols(id, y, X, invalid_field_value)

# # use only rows that have no incomplete values
# (id, y, X) = clean_data.full_rows(id, y, X, invalid_field_value)

# use only cols that have no incomplete values
# (id, y, X) = clean_data.full_cols(id, y, X, invalid_field_value)

################################################################################
#                      standardize & polynomial expansion                      #
################################################################################
X = polynomial_expansion.polynomial_expansion(X, degree)
(X, _, _) = helpers.standardize(X)
X = helpers.add_offset_parameter(X)

################################################################################
#                               search for lambda                              #
################################################################################
# processes a specific lambda through the cross-validation pipeline and writes
# results to a pickled file for further analysis.
def process_lambda_grid_search(id, y, X, fold_count, seed, gd_func, max_iters, gamma, lamb, degree):
    N, D = X.shape
    initial_w = np.ones(D)

    # k-fold cross-validation
    (w_stars, train_correct_ratios, test_correct_ratios) = \
        cross_validation.cross_validate(id, y, X, fold_count, seed, gd_func,
                                        initial_w, max_iters, gamma, lamb)

    filename = "train_clean_avg_L2_degree{degree}_fold{fold}_gamma{gamma}_iter{iter}_lamb{lamb}.pickle".format(degree=degree, fold=fold_count, gamma=gamma, iter=max_iters, lamb=lamb)
    with open(filename, "wb") as pickle_file:
        pickle.dump((w_stars, train_correct_ratios, test_correct_ratios), pickle_file)

# process cross validation in parallel
def process_fold(y_train, X_train, y_test, X_test, initial_w, max_iters, gamma, lamb, foldId):
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

    filename = "jan_train_clean_avg_L2_degree{degree}_fold{fold}_foldId{foldId}_gamma{gamma}_iter{iter}_lamb{lamb}.pickle".format(degree=degree, fold=fold_count, foldId=foldId, gamma=gamma, iter=max_iters, lamb=lamb)
    with open(filename, "wb") as pickle_file:
        pickle.dump((w_star, train_correct_ratio, test_correct_ratio), pickle_file)

    return w_star, train_correct_ratio, test_correct_ratio


################################################################################
#                              save data for analysis                          #
################################################################################

# parallel process different lambdas and write to files
from joblib import Parallel, delayed

# # lambda grid search parallel
# Parallel(n_jobs=-1, verbose=11)(delayed(process_lambda_grid_search)(id, y, X, fold_count, seed, gd_func, max_iters, gamma, lamb, degree) for lamb in lambdas)

# cross-validation parallel
N, D = X.shape
# initial_w = np.ones(D)
initial_w = np.random.uniform(-0.001, 0.001, D)
Parallel(n_jobs=1, verbose=11)(delayed(process_fold)(y_train, X_train, y_test, X_test, initial_w, max_iters, gamma, lambda_best, foldId) for (foldId, (id_train, y_train, X_train, id_test, y_test, X_test)) in enumerate(cross_validation.cross_validate_generator(id, y, X, fold_count, seed)))
