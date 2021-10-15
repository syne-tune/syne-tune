"""
Entrypoint script that allows to launch a tuning job remotely.
It loads the tuner from a specified path then runs it.
"""
import logging
from argparse import ArgumentParser
from pathlib import Path

from sagemaker_tune.tuner import Tuner


def get_tuner_callbacks(tuner: Tuner):
    from sagemaker_tune.backend.simulator_backend.simulator_callback import \
        create_simulator_callback
    from sagemaker_tune.backend.simulator_backend.simulator_callback import \
        SimulatorBackend

    backend = tuner.backend
    if isinstance(backend, SimulatorBackend):
        simulator_callback = create_simulator_callback(tuner)
        return [simulator_callback]
    else:
        return None


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tuner_path', type=str, default="tuner/")
    parser.add_argument('--store_logs', dest='store_logs', action='store_true', default=False)
    parser.add_argument('--log_level', type=int, default=logging.INFO)
    parser.add_argument('--no_tuner_logging', type=str, default='False')
    args, _ = parser.parse_known_args()

    root = logging.getLogger()
    root.setLevel(args.log_level)

    tuner_path = Path(args.tuner_path)
    logging.info(f"load tuner from path {args.tuner_path}")
    tuner = Tuner.load(tuner_path)

    if args.store_logs:
        # inform the backend of the desired path so that logs are persisted
        tuner.backend.set_path(results_root='/opt/ml/checkpoints', tuner_name=tuner.name)

    # Run the tuner on the sagemaker instance. If the simulation back-end is
    # used, this needs a specific callback
    tuner_callbacks = get_tuner_callbacks(tuner)
    if args.no_tuner_logging == 'True':
        logging.getLogger('sagemaker_tune.tuner').setLevel(logging.ERROR)
    logging.info("starting remote tuning")
    tuner.run(callbacks=tuner_callbacks)
