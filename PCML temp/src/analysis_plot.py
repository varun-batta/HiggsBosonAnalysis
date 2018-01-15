import matplotlib.pyplot as plt
import numpy as np

import clean_data
import helpers
import os
import pickle
import polynomial_expansion
import proj1_helpers
import re


def plot_cross_validation_lambda_accuraccy(lambdas, avg_train_error_ratios, avg_test_error_ratios):
    fig = plt.figure()

    plt.semilogx()
    plt.grid()

    plt.plot(lambdas, avg_train_error_ratios, color="b", marker=".", label="Training")
    plt.hold('on')
    plt.plot(lambdas, avg_test_error_ratios, color='r', marker=".", label="Validation")
    plt.hold('off')

    # plt.title("Misclassification ratio as a function of lambda")
    plt.xlabel('lambda')
    plt.ylabel("Misclassification ratio")

    ax = fig.add_subplot(111)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles, loc="best")

    plt.show()


def read_data_lambda_grid_search(dirname, regularization_level, degree, iter_count, fold):
    lambdas = []
    lamb_wStars_dict = {}
    lamb_trainRatios_dict = {}
    lamb_testRatios_dict = {}

    os.chdir(dirname)
    for filename in os.listdir("."):
        # regularization level match
        match = re.search(r"_L(\d)_degree", filename)
        regularization_level_file = int(match.group(1))

        # degree match
        match = re.search(r"_degree(\d)_", filename)
        degree_file = int(match.group(1))

        # iter match
        match = re.search(r"_iter(\d+)_", filename)
        iter_count_file = int(match.group(1))

        # fold match
        match = re.search(r"_fold(\d)_", filename)
        fold_file = int(match.group(1))

        if regularization_level_file == regularization_level and degree_file == degree and iter_count_file == iter_count and fold_file == fold:

            # extract lambda from filename
            match = re.search(r"_lamb(\d+\.\d+)\.pickle", filename)
            lamb = float(match.group(1))

            # open and read file
            with open(filename, "rb") as pickle_file:
                (w_stars, train_correct_ratios, test_correct_ratios) = pickle.load(pickle_file)

                lambdas.append(lamb)
                lamb_wStars_dict[lamb] = w_stars
                lamb_trainRatios_dict[lamb] = train_correct_ratios
                lamb_testRatios_dict[lamb] = test_correct_ratios

    lambdas = sorted(lambdas)

    return (lambdas, lamb_wStars_dict, lamb_trainRatios_dict, lamb_testRatios_dict)


def read_data_lambda_cross_validation(dirname):
    w_stars = []
    train_ratios = []
    test_ratios = []

    os.chdir(dirname)
    for filename in os.listdir("."):

        # open and read file
        with open(filename, "rb") as pickle_file:
            (w_star, train_correct_ratio, test_correct_ratio) = pickle.load(pickle_file)

            w_stars.append(w_star)
            train_ratios.append(train_correct_ratio)
            test_ratios.append(test_correct_ratio)

    os.chdir("..")
    return (w_stars, train_ratios, test_ratios)


################################################################################

# grid search for lambda plot
(lambdas, lamb_wStars_dict, lamb_trainRatios_dict, lamb_testRatios_dict) = read_data_lambda_grid_search("train_clean_avg_L2_degree2_fold3_gamma0.05_iter1000/", 2, 2, 1000, 3)
avg_train_correct_ratios = np.array([np.mean(lamb_trainRatios_dict[lamb]) for lamb in lambdas])
avg_test_correct_ratios = np.array([np.mean(lamb_testRatios_dict[lamb]) for lamb in lambdas])
avg_train_error_ratios = 1 - avg_train_correct_ratios
avg_test_error_ratios = 1 - avg_test_correct_ratios

plot_cross_validation_lambda_accuraccy(lambdas, avg_train_error_ratios, avg_test_error_ratios)

# # get best w_star for given lambda
# (w_stars_5, train_ratios_5, test_ratios_5) = read_data_lambda_cross_validation("train_clean_avg_L2_degree2_fold5_gamma0.05_iter5000_lambda10.6")
# (w_stars_8, train_ratios_8, test_ratios_8) = read_data_lambda_cross_validation("train_clean_avg_L2_degree2_fold8_gamma0.05_iter10000_lambda10.6")
# # (w_stars_jan, train_ratios_jan, test_ratios_jan) = read_data_lambda_cross_validation("jan_train_clean_avg_L2_degree2_fold1_foldId0_gamma0.04_iter5000_lamb0")
# (w_stars_jan, train_ratios_jan, test_ratios_jan) = read_data_lambda_cross_validation("jan_train_clean_avg_L2_degree2_fold1_foldId0_gamma0.08_iter7000_lamb0")
#
# ################################################################################
# #                                   submission                                 #
# ################################################################################
# w_stars = w_stars_jan
# print(train_ratios_jan)
#
# ################################################################################
# #                                    read data                                 #
# ################################################################################
# DATA_TEST_PATH = "../data/test_clean_avg.csv"
# (y, X, id) = proj1_helpers.load_csv_data(DATA_TEST_PATH, sub_sample=False)
#
# # y is categorical, so we want integers (-1, 1) instead of floats (-1.0, 1.0)
# # Modified here instead of in load_csv_data, because we don't know if we have the
# # right to change the provided functions.
# y = y.astype(int)
#
# # The formulas used for the cost and gradients of the logistic function expect
# # categories that are 0/1 for some terms to disappear in the equations.
# y[np.where(y == -1)] = 0
#
# ################################################################################
# #                                   clean data                                 #
# ################################################################################
# # one-hot coding for "PRI_jet_num" (column 22)
# (id, y, X) = clean_data.one_hot_PRI_jet_num(id, y, X)
#
# ################################################################################
# #                      standardize & polynomial expansion                      #
# ################################################################################
# degree = 2
# X = polynomial_expansion.polynomial_expansion(X, degree)
# (X, _, _) = helpers.standardize(X)
# X = helpers.add_offset_parameter(X)
#
# N, D = X.shape
#
# ################################################################################
# #                                  predict labels                              #
# ################################################################################
# # compute labels for each w_star obtained through cross validation
# y_preds = np.zeros((N, len(w_stars)))
# for idx, w_star in enumerate(w_stars):
#     y_preds[:, idx] = proj1_helpers.predict_labels(w_star, X)
#
# # majority voting for best y
# y_pred_final = np.zeros(N)
# for i in range(N):
#     if np.sum(y_preds[i, :] > 0):
#         y_pred_final[i] = 1
#     else:
#         y_pred_final[i] = -1
#
# proj1_helpers.create_csv_submission(id, y_pred_final, "submission_jan_test_clean_avg_L2_degree2_fold1_foldId0_gamma0.08_iter7000_lamb0.csv")
#
# print("done")
