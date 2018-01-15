# -*- coding: utf-8 -*-
"""
Data cleaning functions.
"""

import numpy as np
import math


def split_categorical(id, y, X):
    """
    Separates data according to their categorical label.

    Parameters
    ==========
    id: np.ndarray
        N-vector of sample identifiers.

    y: np.ndarray
        N-vector of sample labels.

    X: np.ndarray
        N x D matrix of N samples with D features each.

    Returns
    =======
    id_0: np.ndarray
        A-vector of sample identifiers for category 0.

    y_0: np.ndarray
        A-vector of sample labels for category 0.

    X_0: np.ndarray
        A x D matrix of A samples with D features each for category 0.

    id_1: np.ndarray
        B-vector of sample identifiers for category 1.

    y_1: np.ndarray
        B-vector of sample labels for category 1.

    X_1: np.ndarray
        B x D matrix of B samples with D features each for category 1.
    """

    category_0_condition = y == 1
    category_1_condition = np.logical_not(category_0_condition)

    id_0 = id[category_0_condition]
    id_1 = id[category_1_condition]

    y_0 = y[category_0_condition]
    y_1 = y[category_1_condition]

    X_0 = X[category_0_condition]
    X_1 = X[category_1_condition]

    return (id_0, y_0, X_0, id_1, y_1, X_1)


def split_complete_and_incomplete(id, y, X, incomplete_field_value):
    """
    Separates complete entries from incomplete ones.

    Parameters
    ==========
    id: np.ndarray
        N-vector of sample identifiers.

    y: np.ndarray
        N-vector of sample labels.

    X: np.ndarray
        N x D matrix of N samples with D features each.

    incomplete_field_value: int/np.float64
        Value that is used to detect if an entry is incomplete (ex. -999)

    Returns
    =======
    id_complete: np.ndarray
        C-vector of C complete sample identifiers.

    y_complete: np.ndarray
        C-vector of C complete sample labels.

    X_complete: np.ndarray
        C x D matrix of C complete samples with D features each.

    id_incomplete: np.ndarray
        I-vector of I incomplete sample identifiers.

    y_incomplete: np.ndarray
        I-vector of I incomplete sample labels.

    X_incomplete: np.ndarray
        I x D matrix of I incomplete samples with D features each.
    """

    # mask of rows that have (some) incomplete fields and (all) complete fields
    incomplete_condition = np.any(X == incomplete_field_value, axis=1)
    complete_condition = np.logical_not(incomplete_condition)

    # separate rows that have incomplete and complete fields
    id_incomplete = id[incomplete_condition]
    id_complete = id[complete_condition]

    y_incomplete = y[incomplete_condition]
    y_complete = y[complete_condition]

    X_incomplete = X[incomplete_condition]
    X_complete = X[complete_condition]

    return (id_complete, y_complete, X_complete, id_incomplete, y_incomplete, X_incomplete)


def replace_incomplete_fields_with_closest_complete_fields(complete_rows,
                                                           incomplete_rows,
                                                           incomplete_field_value):
    """
    Replaces incomplete fields in the dataset with the corresponding fields
    from the closest complete dataset.

    Parameters
    ==========
    complete_rows: np.ndarray
        C x D matrix of C complete samples with D features each.

    incomplete_rows: np.ndarray
        I x D matrix of I incomplete samples with D features each.

    incomplete_field_value: int/np.float64
        Value that is used to detect if an entry is incomplete (ex. -999)

    Returns
    =======
    filled_incomplete_rows: np.ndarray
        I x D matrix of I reconstructed complete samples with D features each.
    """

    # Placeholder for filled-in incomplete rows
    filled_incomplete_rows = np.copy(incomplete_rows)

    onePctOfDataF = incomplete_rows.shape[0] / 100.0
    onePctOfDataI = math.ceil(onePctOfDataF)

    # Compute distance of each incomplete row with all complete rows,
    # and find minimum one.
    for (incomplete_row_idx, incomplete_row) in enumerate(incomplete_rows):
        # Textual progress feedback
        if (incomplete_row_idx % onePctOfDataI) == 0:
            print("cleaning data: {pctDone} % done".format(pctDone=(int(incomplete_row_idx / onePctOfDataF))))

        # Mask for defining valid and invalid fields within a row.
        valid_fields_condition = incomplete_row != incomplete_field_value
        invalid_fields_condition = np.logical_not(valid_fields_condition)

        # Remove incomplete fields to avoid influencing the distance computation.
        # Make valid_incomplete_rows 1 x D (less than D in reality since we removed
        # fields) so we can later broadcast to C x D and do sum of squared differences.
        valid_incomplete_row = incomplete_row[valid_fields_condition]
        valid_incomplete_rows = valid_incomplete_row[np.newaxis, :]
        valid_complete_rows = complete_rows[:, valid_fields_condition]

        # Compute distances and select smallest one (index in complete_rows).
        distances = np.sqrt(np.sum((valid_complete_rows - valid_incomplete_rows)**2, axis=1))
        min_distance_idx = np.argmin(distances)
        # print("min_distance =", distances[min_distance_idx])

        # Fill-in missing fields in incomplete rows with corresponding fields in
        # closest complete row.
        closest_complete_row = complete_rows[min_distance_idx]
        filled_incomplete_rows[incomplete_row_idx, invalid_fields_condition] = \
            closest_complete_row[invalid_fields_condition]

    return filled_incomplete_rows


def nearest_neighbor(id, y, X, incomplete_field_value):
    """
    Cleans the dataset by replacing incomplete fields with the
    corresponding fields from the closest data entry.

    Parameters
    ==========
    id: np.ndarray
        N-vector of sample identifiers.

    y: np.ndarray
        N-vector of sample labels.

    X: np.ndarray
        N x D matrix of N samples with D features each.

    incomplete_field_value: int/np.float64
        Value that is used to detect if an entry is incomplete (ex. -999)

    Returns
    =======
    id_clean: np.ndarray
        N-vector of identifiers.

    y_clean: np.ndarray
        N-vector of labels.

    X_clean: np.ndarray
        N x D matrix of N samples with D features each. All incomplete fields are
        filled with the corresponding fields from the closest data entry with the
        same label.
    """

    # copy the dataset because we are going to modify them (want to maintaint
    # "functional" programming-style interface.
    id = np.copy(id)
    y = np.copy(y)
    X = np.copy(X)

    (id_0, y_0, X_0, id_1, y_1, X_1) = split_categorical(id, y, X)

    (id_0_complete, y_0_complete, X_0_complete, id_0_incomplete, y_0_incomplete, X_0_incomplete) = \
        split_complete_and_incomplete(id_0, y_0, X_0, incomplete_field_value)

    (id_1_complete, y_1_complete, X_1_complete, id_1_incomplete, y_1_incomplete, X_1_incomplete) = \
        split_complete_and_incomplete(id_1, y_1, X_1, incomplete_field_value)

    X_0_incomplete_filled = \
        replace_incomplete_fields_with_closest_complete_fields(X_0_complete,
                                                               X_0_incomplete,
                                                               incomplete_field_value)

    X_1_incomplete_filled = \
        replace_incomplete_fields_with_closest_complete_fields(X_1_complete,
                                                               X_1_incomplete,
                                                               incomplete_field_value)

    # stick data back into input shape
    id_all = np.hstack((id_0_complete, id_0_incomplete, id_1_complete, id_1_incomplete))
    y_all = np.hstack((y_0_complete, y_0_incomplete, y_1_complete, y_1_incomplete))
    X_all = np.vstack((X_0_complete, X_0_incomplete_filled, X_1_complete, X_1_incomplete_filled))

    # return same data ordering as in input
    order = np.argsort(id_all)
    id_clean = id_all[order]
    y_clean = y_all[order]
    X_clean = X_all[order]

    return (id_clean, y_clean, X_clean)


def full_rows(id, y, X, incomplete_field_value):
    """
    Cleans the dataset by selecting rows that have all their dimensions
    populated with valid data.

    Parameters
    ==========
    id: np.ndarray
        N-vector of sample identifiers.

    y: np.ndarray
        N-vector of sample labels.

    X: np.ndarray
        N x D matrix of N samples with D features each.

    incomplete_field_value: int/np.float64
        Value that is used to detect if an entry is incomplete (ex. -999)

    Returns
    =======
    id_clean: np.ndarray
        N-vector of identifiers.

    y_clean: np.ndarray
        N-vector of labels.

    X_clean: np.ndarray
        M x D matrix of M samples with D features each. M corresponds to the
        number of rows that have no incomplete values in any of their fields.
    """


    complete_row_condition = np.all(X != incomplete_field_value, axis=1)

    id_clean = id[complete_row_condition]
    y_clean = y[complete_row_condition]
    X_clean = X[complete_row_condition]

    return (id_clean, y_clean, X_clean)


def full_cols(id, y, X, incomplete_field_value):
    """
    Cleans the dataset by selecting cols that have all their dimensions
    populated with valid data.

    Parameters
    ==========
    id: np.ndarray
        N-vector of sample identifiers.

    y: np.ndarray
        N-vector of sample labels.

    X: np.ndarray
        N x D matrix of N samples with D features each.

    incomplete_field_value: int/np.float64
        Value that is used to detect if an entry is incomplete (ex. -999)

    Returns
    =======
    id_clean: np.ndarray
        N-vector of identifiers.

    y_clean: np.ndarray
        N-vector of labels.

    X_clean: np.ndarray
        N x E matrix of N samples with E features each. E corresponds to the
        number of cols that have no incomplete values in any of their fields.
    """

    complete_col_condition = np.all(X != incomplete_field_value, axis=0)

    id_clean = id
    y_clean = y
    X_clean = X[:, complete_col_condition]

    return (id_clean, y_clean, X_clean)

def keep_dimensions(X, inds):
    """
    Extracts the columns given by 'inds' from the X dataset matrix a returns
    the resulting matrix.

    Parameters
    ==========
    X: np.ndarray
        N x D matrix of N samples with D features each.

    inds: np.ndarray
        K-vector of column indices (to extract)

    Returns
    =======
    X_clean: np.ndarray
        N x K matrix of N samples with K features each.
    """

    X_clean = X[:, inds]

    return X_clean


def one_hot_PRI_jet_num(id, y, X):
    """
    Performs one-hot coding for the "PRI_jet_num" categorical column.

    Parameters
    ==========
    id: np.ndarray
        N-vector of sample identifiers.

    y: np.ndarray
        N-vector of sample labels.

    X: np.ndarray
        N x D matrix of N samples with D features each.

    Returns
    =======
    id_clean: np.ndarray
        N-vector of identifiers.

    y_clean: np.ndarray
        N-vector of labels.

    X_clean: np.ndarray
        N x (D + 4) matrix of N samples with (D + 4) features each. The
        "PRI_jet_num" parameter has been one-hot coded into 4 parameters.
    """

    N, D = X.shape

    # one-hot coding for "PRI_jet_num" (column 22)
    PRI_jet_num = X[:, 22].astype(int)
    PRI_jet_num_categories = np.unique(PRI_jet_num)
    PRI_jet_num_one_hot = np.zeros((N, len(PRI_jet_num_categories)))
    PRI_jet_num_one_hot[np.arange(N, dtype=int), PRI_jet_num] = 1

    X_without_PRI_jet_num = X[:, np.r_[np.arange(22, dtype=int), np.arange(23, D, dtype=int)]]

    id_clean = id
    y_clean = y
    X_clean = np.hstack((X_without_PRI_jet_num, PRI_jet_num_one_hot))

    return (id_clean, y_clean, X_clean)


def avg_incomplete_cols(id, y, X, incomplete_field_value):
    """
    Replaces all incomplete values with the average of the corresponding column
    (excluding other incomplete values).

    Parameters
    ==========
    id: np.ndarray
        N-vector of sample identifiers.

    y: np.ndarray
        N-vector of sample labels.

    X: np.ndarray
        N x D matrix of N samples with D features each.

    Returns
    =======
    id_clean: np.ndarray
        N-vector of identifiers.

    y_clean: np.ndarray
        N-vector of labels.

    X_clean: np.ndarray
        N x D matrix of N samples with D features each. The incomplete values
        are replaced with the average value of the corresponding column
        (excluding other incomplete values)
    """

    # copy the dataset because we are going to modify them (want to maintaint
    # "functional" programming-style interface.
    X_clean = np.copy(X)

    for idx, col in enumerate(X.T):
        X_clean[:, idx][col == incomplete_field_value] = np.mean(col[col != incomplete_field_value])

    id_clean = id
    y_clean = y

    return (id_clean, y_clean, X_clean)


def outliers_to_col_avg(X_train, X_test, outlier=-999):
    """
    For each column substitute outlier value by the average of that column.

    Parameters
    ==========
    X_train: np.ndarray
        N x D matrix of N train samples with D features each.

    X_test: np.ndarray
        N x D matrix of N test samples with D features each.

    Returns
    =======
    X_train_clean: np.ndarray
        N x D matrix of N samples with D features each. The incomplete values
        are replaced with the average value of the corresponding column
        (excluding other incomplete values)

    X_test_clean: np.ndarray
        N x D matrix of N samples with D features each. The incomplete values
        are replaced with the average value of the corresponding column
        (excluding other incomplete values)
    """

    N_train, D_train = X_train.shape

    # Stack the training and test data into onde matrix.
    X = np.vstack((X_train, X_test))

    for idx, col in enumerate(X.T):
        X[:, idx][col == outlier] = np.mean(col[col != outlier])

    X_train_clean = X[:N_train]
    X_test_clean  = X[N_train:]

    return X_train_clean, X_test_clean
