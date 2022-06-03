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
from autograd import numpy as np

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon import (
    Parameter,
    ParameterDict,
    Block,
)


def test_parameter():
    p = Parameter(name="abc", shape=(1,))
    p.initialize()
    data = p.data
    grad = p.grad


def test_parameter_dict():
    pd = ParameterDict("pd")
    pd.initialize()
    p = pd.get("def")


def test_block():
    class TestBlock(Block):
        def __init__(self):
            super(TestBlock, self).__init__()
            with self.name_scope():
                self.a = self.params.get("a", shape=(10,))
                self.b = self.params.get("b", shape=(10,))

        def forward(self, x):
            return x + self.a.data() + self.b.data()

    t = TestBlock()
    t.initialize()
    print(t.a.grad_req)
    t.a.set_data(np.ones((10,)))
    assert "a" in t.a.name
    x = np.zeros((10,))
    y = t(x)
    print(y)
