# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import numpy as np
from xgboost import XGBClassifier
from sklearn import preprocessing


def get_data_splits(seed, X, Y, test_threshold, smallest_train):
    np.random.seed(seed)
    shuffled_order = list(range(len(Y)))
    num_classes = len(np.unique(Y))
    while True:
        np.random.shuffle(shuffled_order)
        X_shuffled = np.array([X[ii] for ii in shuffled_order])
        Y_shuffled = np.array([Y[ii] for ii in shuffled_order])
        # Make sure all classes are represented
        if len(np.unique(Y_shuffled[:smallest_train])) == num_classes:
            break

    X_train_raw = X_shuffled[:test_threshold, :]
    X_test_raw = X_shuffled[test_threshold:, :]
    Y_train = Y_shuffled[:test_threshold]
    Y_test = Y_shuffled[test_threshold:]

    scaler = preprocessing.StandardScaler().fit(X_train_raw)

    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, Y_train, X_test, Y_test


def get_log_lik(model, X, Y):
    return np.sum([model.predict_log_proba(X)[kk, Y[kk]] for kk in range(len(Y))])


def get_error_count(model, X, Y):
    return np.sum(model.predict(X) != Y)


def train_and_evaluate(
    X_train, Y_train, par_dict, X_test, Y_test, model_name, llik=True, seed=0
):
    if model_name == "XGBClassifier":
        max_depth = par_dict["max_depth"] if "max_depth" in par_dict else None
        learning_rate = (
            par_dict["learning_rate"] if "learning_rate" in par_dict else None
        )
        reg_alpha = par_dict["reg_alpha"] if "reg_alpha" in par_dict else None
        min_child_weight = (
            par_dict["min_child_weight"] if "min_child_weight" in par_dict else None
        )
        reg_lambda = par_dict["reg_lambda"] if "reg_lambda" in par_dict else None
        n_estimators = par_dict["n_estimators"] if "n_estimators" in par_dict else None

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            objective="binary:logistic",
            seed=seed,
        )
    else:
        raise ValueError
    model.fit(X_train, Y_train)

    if llik:
        test_log_lik = get_log_lik(model, X_test, Y_test)
        train_log_lik = get_log_lik(model, X_train, Y_train)
    else:
        test_log_lik, train_log_lik = None, None
    test_error = get_error_count(model, X_test, Y_test)
    train_error = get_error_count(model, X_train, Y_train)

    return test_error, train_error, test_log_lik, train_log_lik


def make_sample(samp_type, min_lim, max_lim, scale="log"):
    if scale == "log":
        samp_min = np.log(min_lim)
        if samp_type == "integer":
            samp_max = np.log(max_lim + 1)
        else:
            samp_max = np.log(max_lim)
    else:
        raise ValueError

    samp_raw = np.random.uniform(samp_min, samp_max)

    if scale == "log":
        samp_cont = np.exp(samp_raw)
    else:
        samp_cont = samp_raw

    if samp_type == "continuous":
        return samp_cont
    elif samp_type == "integer":
        return int(np.floor(samp_cont))
