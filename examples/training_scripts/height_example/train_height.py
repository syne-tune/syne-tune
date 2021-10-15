"""
Example similar to Raytune, https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/skopt_example.py
"""
import logging
import time

from sagemaker_tune.report import Reporter
from argparse import ArgumentParser


report = Reporter()


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--steps', type=int)
    parser.add_argument('--width', type=float)
    parser.add_argument('--height', type=float)
    parser.add_argument('--sleep_time', type=float, default=0.1)

    args, _ = parser.parse_known_args()

    width = args.width
    height = args.height
    for step in range(args.steps):
        dummy_score = (0.1 + width * step / 100) ** (-1) + height * 0.1
        # Feed the score back to Sagemaker Tune.
        report(step=step, mean_loss=dummy_score, epoch=step + 1)
        time.sleep(args.sleep_time)
