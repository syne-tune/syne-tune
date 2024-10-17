import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import xgboost
from sklearn.datasets import load_digits

from syne_tune import Reporter
from syne_tune.constants import ST_CHECKPOINT_DIR


class SyneTuneCallback(xgboost.callback.TrainingCallback):
    def __init__(self, error_metric: str) -> None:
        self.reporter = Reporter()
        self.error_metric = error_metric

    def after_iteration(self, model, epoch, evals_log):
        metrics = list(evals_log.values())[-1][self.error_metric]
        self.reporter(**{self.error_metric: metrics[-1]})
        pass


def train(
    checkpoint_dir: str,
    n_estimators: int,
    max_depth: int,
    gamma: float,
    reg_lambda: float,
    early_stopping_rounds: int = 5,
) -> None:
    eval_metric = "merror"
    early_stop = xgboost.callback.EarlyStopping(
        rounds=early_stopping_rounds, save_best=True
    )
    X, y = load_digits(return_X_y=True)

    clf = xgboost.XGBClassifier(
        n_estimators=n_estimators,
        reg_lambda=reg_lambda,
        gamma=gamma,
        max_depth=max_depth,
        eval_metric=eval_metric,
        callbacks=[early_stop, SyneTuneCallback(error_metric=eval_metric)],
    )
    clf.fit(
        X,
        y,
        eval_set=[(X, y)],
    )
    print("Total boosted rounds:", clf.get_booster().num_boosted_rounds())

    save_model(clf, checkpoint_dir=checkpoint_dir)


def save_model(clf, checkpoint_dir):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = os.path.join(checkpoint_dir, "model.json")
    clf.save_model(path)


def load_model(checkpoint_dir):
    path = os.path.join(checkpoint_dir, "model.json")
    loaded = xgboost.XGBClassifier()
    loaded.load_model(path)
    return loaded


def evaluate_accuracy(checkpoint_dir):
    X, y = load_digits(return_X_y=True)

    clf = load_model(checkpoint_dir=checkpoint_dir)
    y_pred = clf.predict(X)
    return (np.equal(y, y_pred) * 1.0).mean()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--max_depth", type=int, required=False, default=1)
    parser.add_argument("--gamma", type=float, required=False, default=2)
    parser.add_argument("--reg_lambda", type=float, required=False, default=0.001)
    parser.add_argument("--n_estimators", type=int, required=False, default=10)
    parser.add_argument(f"--{ST_CHECKPOINT_DIR}", type=str, default="./")

    args, _ = parser.parse_known_args()

    checkpoint_dir = Path(vars(args)[ST_CHECKPOINT_DIR])

    train(
        checkpoint_dir=checkpoint_dir,
        max_depth=args.max_depth,
        gamma=args.gamma,
        reg_lambda=args.reg_lambda,
        n_estimators=args.n_estimators,
    )
