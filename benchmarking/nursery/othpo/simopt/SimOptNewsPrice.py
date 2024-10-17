from argparse import ArgumentParser
import logging
from mrg32k3a.mrg32k3a import MRG32k3a
import numpy as np
import pickle
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parent))

from simopt.models.dynamnews import DynamNewsMaxProfit

from syne_tune import Reporter

from simopt_helpers import evaluate_problem_price

if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("--price_A", type=float)
    parser.add_argument("--price_B", type=float)
    parser.add_argument("--price_C", type=float)
    parser.add_argument("--time_idx", type=int)

    args, _ = parser.parse_known_args()
    report = Reporter()

    rand_gen = MRG32k3a(s_ss_sss_index=[1, 2, 3])

    file_dir = Path(__file__).parent / "generated_files"

    args.factors_file = str(file_dir) + "/default_fixed_factors.p"
    args.context_file = (
        str(file_dir) + "/opt-price-random-walk-utility-context-default.p"
    )

    model_fixed_factors = pickle.load(open(args.factors_file, "rb"))
    context_matrices = pickle.load(open(args.context_file, "rb"))

    # Update factors from default using the context for the given time stamp
    model_fixed_factors["c_utility"] = context_matrices["c_utility"][args.time_idx, :]
    model_fixed_factors["init_level"] = context_matrices["init_level"][args.time_idx, :]

    dyn_news_prob = DynamNewsMaxProfit(model_fixed_factors=model_fixed_factors)

    prices = np.array([args.price_A, args.price_B, args.price_C])
    sim_score = evaluate_problem_price(dyn_news_prob, prices, rand_gen, reps=10)

    # Feed the score back to Syne Tune.
    report(profit=sim_score, epoch=1)
