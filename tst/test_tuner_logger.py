

import io
from pathlib import Path
from unittest.mock import patch

from syne_tune.tuner_logger import TunerLogger, Colors
from syne_tune.tuning_status import TuningStatus


@patch('time.strftime', return_value="[12:00:00]")
def test_tuner_logger_experiment_header(mock_strftime):
    with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
        logger = TunerLogger(use_colors=False, use_emojis=False)
        logger.print_experiment_header(
            name="test-experiment",
            backend_name="local",
            n_workers=4,
            scheduler_name="fifo",
            results_path=Path("/tmp/results"),
            log_path="/tmp/logs",
            metric_names=["accuracy"],
            metric_mode="max",
            stop_criterion_info="stop at 100",
            config_space={"lr": "uniform(0.1, 1.0)"}
        )
        output = mock_stdout.getvalue()
        expected_output = (
            "\nSyne Tune - Hyperparameter Optimization\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "\nExperiment Configuration\n"
            "├─ Name: test-experiment\n"
            "├─ Backend: local\n"
            "├─ Workers: 4\n"
            "├─ Scheduler: fifo\n"
            "├─ Results Path: /tmp/results\n"
            "└─ Log Path: /tmp/logs\n"
            "\nOptimization Target\n"
            "├─ Metric: accuracy\n"
            "├─ Mode: max\n"
            "└─ Stop Criterion: stop at 100\n"
            "\nSearch Space\n"
            "└─ lr: uniform(0.1, 1.0)\n"
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        )
        assert output == expected_output.replace("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "━" * 80)


@patch('time.strftime', return_value="[12:00:00]")
def test_tuner_logger_colors_and_emojis(mock_strftime):
    with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
        logger = TunerLogger(use_colors=True, use_emojis=True)
        logger.print_tuning_start()
        output = mock_stdout.getvalue()
        assert output == f"🏁 {Colors.GREEN}Starting hyperparameter optimization...{Colors.RESET}\n"


@patch('time.strftime', return_value="[12:00:00]")
def test_tuner_logger_no_colors_no_emojis(mock_strftime):
    with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
        logger = TunerLogger(use_colors=False, use_emojis=False)
        logger.print_tuning_start()
        output = mock_stdout.getvalue()
        assert output == "Starting hyperparameter optimization...\n"


@patch('time.strftime', return_value="[12:00:00]")
def test_tuner_logger_print_trial_result(mock_strftime):
    with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
        logger = TunerLogger(use_colors=False, use_emojis=False)
        logger.print_trial_result(trial_id=1, result={"accuracy": 0.9, "loss": 0.1}, epoch=1, total_epochs=10)
        output = mock_stdout.getvalue()
        assert output == "[12:00:00] Trial 1 | Epoch 1/10 | accuracy: 0.9000 | loss: 0.1000\n"


@patch('time.strftime', return_value="[12:00:00]")
@patch('time.perf_counter', return_value=0.0)
def test_tuner_logger_print_tuning_status(mock_perf_counter, mock_strftime):
    with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
        logger = TunerLogger(use_colors=False, use_emojis=False)
        status = TuningStatus(metric_names=["accuracy"])
        logger.print_tuning_status(status)
        output = mock_stdout.getvalue()

    expected = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tuning Status (last metric is reported)
0 trials running, 0 finished (0 until the end), 0.00s wallclock-time

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
    assert output == expected
