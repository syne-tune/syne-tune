"""
Example demonstrating QuantileRegressionSurrogateModel on artificial data.

This script shows how to:
1. Define a config space
2. Generate random artificial data (features + targets)
3. Fit the QuantileRegressionSurrogateModel with different model types
4. Make predictions and inspect quantile outputs

Supported model types:
- "gradient_boosting": Default, uses sklearn's GradientBoostingRegressor
- "tabpfn": Uses TabPFN 2.5 (requires: pip install tabpfn + HuggingFace auth)
"""

import argparse
import time

import numpy as np
import pandas as pd

from syne_tune.config_space import uniform, randint
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_surrogate import (
    QuantileRegressionSurrogateModel,
)


def generate_artificial_data(config_space: dict, n_samples: int, noise_std: float = 0.1):
    """
    Generate artificial data where y is a function of the hyperparameters.

    The target function is: y = x1 + 0.5 * x2^2 + noise
    """
    random_state = np.random.RandomState(42)

    # Sample random configurations
    configs = []
    for _ in range(n_samples):
        config = {}
        for k, v in config_space.items():
            if hasattr(v, "sample"):
                config[k] = v.sample(random_state=random_state)
            else:
                config[k] = v
        configs.append(config)

    df_features = pd.DataFrame(configs)

    # Create an artificial target: y = x1 + 0.5 * x2^2 + noise
    y = (
        df_features["x1"].values
        + 0.5 * (df_features["x2"].values ** 2)
        + random_state.normal(0, noise_std, size=n_samples)
    )

    return df_features, y


def main(model_type: str = "gradient_boosting"):
    # Define a simple config space with continuous and integer hyperparameters
    config_space = {
        "x1": uniform(0.0, 1.0),
        "x2": uniform(-1.0, 1.0),
        "x3": randint(1, 10),
    }

    # Generate artificial training data
    n_train = 200
    df_train, y_train = generate_artificial_data(config_space, n_train, noise_std=0.1)

    print(f"Model type: {model_type}")
    print(f"Training data shape: {df_train.shape}")
    print(f"Target shape: {y_train.shape}")
    print(f"Target range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print()

    # Create the surrogate model with specified model type
    surrogate = QuantileRegressionSurrogateModel(
        config_space=config_space,
        mode="min",  # we want to minimize the target
        quantiles=5,  # number of quantiles to estimate
        random_state=np.random.RandomState(123),
        model_type=model_type,
    )

    # Fit the model
    print(f"Fitting QuantileRegressionSurrogateModel (model_type={model_type})...")
    surrogate.fit(df_features=df_train, y=y_train, ncandidates=1000)
    print("Fitting complete!")
    print()

    # Generate test data for predictions
    n_test = 10
    df_test, y_test = generate_artificial_data(config_space, n_test, noise_std=0.1)

    # Make predictions
    predictions = surrogate.predict(df_test)

    print(f"Predictions type: {type(predictions).__name__}")
    print(f"Number of quantiles: {predictions.nquantiles}")
    print(f"Quantile values: {predictions.quantiles}")
    print(f"Results shape: {predictions.results_stacked.shape}")
    print()

    # Show predictions for each quantile
    print("Predictions for each quantile (first 5 test samples):")
    print("-" * 60)
    for i, q in enumerate(predictions.quantiles):
        q_preds = predictions.results(q)[:5]
        print(f"  Quantile {q:.2f}: {q_preds}")
    print()

    # Compare median predictions vs actual values
    median_preds = predictions.mean()
    print("Comparison of median predictions vs actual (first 5 samples):")
    print("-" * 60)
    for i in range(min(5, len(y_test))):
        print(f"  Sample {i}: predicted={median_preds[i]:.3f}, actual={y_test[i]:.3f}")
    print()

    # Demonstrate the suggest functionality
    print("Suggesting best configuration from candidates...")
    suggest_start = time.perf_counter()
    suggested_config = surrogate.suggest(replace_config=True)
    suggest_runtime = time.perf_counter() - suggest_start
    print(f"Suggested config: {suggested_config}")
    print(f"Suggestion runtime: {suggest_runtime:.4f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test QuantileRegressionSurrogateModel with different backends"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="tabicl",
        choices=["gradient_boosting", "tabpfn", "tabicl"],
        help="Model type to use: 'gradient_boosting', 'tabpfn', or 'tabicl'",
    )
    args = parser.parse_args()
    main(model_type=args.model_type)
