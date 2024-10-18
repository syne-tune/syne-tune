import logging
import time

from syne_tune import Reporter
from argparse import ArgumentParser

if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--width", type=float)
    parser.add_argument("--height", type=float)
    args, _ = parser.parse_known_args()
    report = Reporter()

    for step in range(args.epochs):
        time.sleep(0.1)
        dummy_score = 1.0 / (0.1 + args.width * step / 100) + args.height * 0.1
        # Feed the score back to Syne Tune
        report(epoch=step + 1, mean_loss=dummy_score)
