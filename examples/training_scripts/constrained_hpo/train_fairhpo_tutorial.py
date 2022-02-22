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
import logging
import numpy as np
import pickle as pkl
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


from syne_tune import Reporter
from argparse import ArgumentParser


report = Reporter()


def statistical_disparity(model, X, Y, groups):
    """
    :param model: the trained model
    :param X: the input dataset with n observations
    :param Y: binary labels associated to the n observations (1 = positive)
    :param groups: a list of n values binary values defining two different subgroups of the populations
    """
    fY = model.predict(X)
    sp = [0, 0]
    sp[0] = float(len([1 for idx, fy in enumerate(fY) if fy == 1 and groups[idx] == 0])) / len(
        [1 for idx, fy in enumerate(fY) if groups[idx] == 0])
    sp[1] = float(len([1 for idx, fy in enumerate(fY) if fy == 1 and groups[idx] == 1])) / len(
        [1 for idx, fy in enumerate(fY) if groups[idx] == 1])
    return abs(sp[0] - sp[1])


if __name__ == '__main__':

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    p = ArgumentParser()
    p.add_argument('--max_depth', default=10, type=int)
    p.add_argument('--min_samples_split', default=0.5, type=float)
    p.add_argument('--criterion', type=str, default='gini')
    p.add_argument('--data_path', type=str, default='../notebooks/fairhpo_data.pickle')
    p.add_argument('--fairness_threshold', default=0.01, type=float)
    p.add_argument('--seed', default=10, type=int)
    args, _ = p.parse_known_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    with open(args.data_path, 'rb') as handle:
        data_dict = pkl.load(handle)

    classifier = RandomForestClassifier(max_depth=args.max_depth,
                                        min_samples_split=args.min_samples_split,
                                        criterion=args.criterion)
    classifier.fit(data_dict['X_train'], data_dict['Y_train'])
    dsp_foreign_worker = statistical_disparity(
        classifier, data_dict['X_test'], data_dict['Y_test'], data_dict['is_foreign'])
    y_pred = classifier.predict(data_dict['X_test'])
    accuracy = accuracy_score(y_pred, data_dict['Y_test'])
    objective_value = accuracy
    constraint_value = dsp_foreign_worker - args.fairness_threshold  # If DSP < fairness threshold,
    # then we consider the model fair
    report(objective=objective_value, my_constraint_metric=constraint_value)
