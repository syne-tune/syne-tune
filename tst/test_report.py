import logging

from syne_tune import Reporter
from syne_tune.report import retrieve
from syne_tune.constants import ST_METRIC_TAG


def test_report_logger():
    report = Reporter()

    logging.getLogger().setLevel(logging.INFO)

    report(train_nll=1.45, time=1.0, step=2)
    report(train_nll=1.2, time=2.0, step=3)

    prefix = "[" + ST_METRIC_TAG + "]: "
    lines = [
        prefix + '{"train_nll": 1.45, "time": 1.0, "step": 2}\n',
        prefix + '{"train_nll": 1.2, "time": 2.0, "step": 3}\n',
    ]
    metrics = retrieve(log_lines=lines)
    print(metrics)
    assert metrics == [
        {"train_nll": 1.45, "time": 1.0, "step": 2},
        {"train_nll": 1.2, "time": 2.0, "step": 3},
    ]
