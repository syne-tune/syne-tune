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
from syne_tune.experiments import load_experiment


if __name__ == "__main__":
    # Replace with name for your experiment:
    # Run:
    #    ls ~/syne-tune/docs-1/MOBSTER-JOINT-0/
    tuner_name = (
        "docs-1/MOBSTER-JOINT-0/docs-1-nas201-cifar10-0-2023-04-15-11-35-31-201"
    )

    tuning_experiment = load_experiment(tuner_name)
    print(tuning_experiment)

    print(f"best result found: {tuning_experiment.best_config()}")

    tuning_experiment.plot()
