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

import GPy


class GPModel:
    def __init__(self, kernel_type: str = "matern52"):
        self.kernel_type = kernel_type

    def fit(self, X, y):
        noise_prior = GPy.priors.Gamma(0.1, 0.1)
        noise_kernel = GPy.kern.White(X.shape[1])
        noise_kernel.set_prior(noise_prior)

        if self.kernel_type == "matern52":
            kern = GPy.kern.Matern52(X.shape[1], ARD=True) + noise_kernel
        elif self.kernel_type == "rbf":
            kern = GPy.kern.RBF(X.shape[1], ARD=True) + noise_kernel

        self.m = GPy.models.GPClassification(X, y[:, None], kernel=kern)
        self.m.optimize()

    def predict_proba(self, X):
        m = self.m.predict(X)[0]
        return m

    def predict(self, X):
        l = np.round(self.m.predict(X)[0])
        return l
