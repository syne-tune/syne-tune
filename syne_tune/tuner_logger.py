from pathlib import Path
import time
from syne_tune.tuning_status import TuningStatus


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"


class TunerLogger:
    """
    Handles all user-facing output for the Tuner.
    This class can be easily extended or replaced to customize output formatting.
    """

    def __init__(self, use_colors: bool = True, use_emojis: bool = True):
        """
        :param use_colors: Whether to use ANSI colors in output
        :param use_emojis: Whether to use emojis in output
        """
        self.use_colors = use_colors
        self.use_emojis = use_emojis

    def _format_message(
        self, emoji: str, message: str, color: str = Colors.RESET
    ) -> str:
        """Format a message with optional emoji and color."""
        prefix = f"{emoji} " if self.use_emojis else ""
        if self.use_colors:
            return f"{prefix}{color}{message}{Colors.RESET}"
        else:
            return f"{prefix}{message}"

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if self.use_colors:
            return f"{color}{text}{Colors.RESET}"
        return text

    def _get_timestamp(self) -> str:
        """Get current timestamp in [HH:MM:SS] format."""
        return time.strftime("[%H:%M:%S]")

    def print_experiment_header(
        self,
        name: str,
        backend_name: str,
        n_workers: int,
        scheduler_name: str,
        results_path: Path,
        log_path: str,
        metric_names: list[str] = None,
        metric_mode: str = None,
        stop_criterion_info: str = None,
        config_space: dict = None,
    ):
        """Print the experiment configuration header."""
        separator = "━" * 80
        syne_tune_msg = "Syne Tune - Hyperparameter Optimization"
        if self.use_emojis:
            syne_tune_msg = "🚀 " + syne_tune_msg
        print(f"\n{self._color(syne_tune_msg, Colors.CYAN + Colors.BOLD)}")
        print(separator)
        experiment_conf_msg = "Experiment Configuration"
        if self.use_emojis:
            experiment_conf_msg = "📋 " + experiment_conf_msg
        print(f"\n{self._color(experiment_conf_msg, Colors.BOLD)}")
        print(f"├─ Name: {self._color(name, Colors.GREEN)}")
        print(f"├─ Backend: {self._color(backend_name, Colors.CYAN)}")
        print(f"├─ Workers: {self._color(str(n_workers), Colors.YELLOW)}")
        print(f"├─ Scheduler: {self._color(scheduler_name, Colors.MAGENTA)}")
        print(f"├─ Results Path: {self._color(str(results_path), Colors.BLUE)}")
        print(f"└─ Log Path: {self._color(log_path, Colors.BLUE)}")

        # Add optimization target if provided
        if metric_names or metric_mode or stop_criterion_info:
            opt_target_msg = "Optimization Target"
            if self.use_emojis:
                opt_target_msg = "🎯 " + opt_target_msg
            print(f"\n{self._color(opt_target_msg, Colors.BOLD)}")
            if metric_names:
                metric_str = (
                    ", ".join(metric_names)
                    if isinstance(metric_names, list)
                    else str(metric_names)
                )
                print(f"├─ Metric: {self._color(metric_str, Colors.CYAN)}")
            if metric_mode:
                print(f"├─ Mode: {self._color(metric_mode, Colors.CYAN)}")
            if stop_criterion_info:
                print(
                    f"└─ Stop Criterion: {self._color(stop_criterion_info, Colors.CYAN)}"
                )

        # Add search space if provided
        if config_space:
            search_space_msg = "Search Space"
            if self.use_emojis:
                search_space_msg = "⚙️  " + search_space_msg
            print(f"\n{self._color(search_space_msg, Colors.BOLD)}")
            items = list(config_space.items())
            for i, (key, value) in enumerate(items):
                prefix = "└─" if i == len(items) - 1 else "├─"
                print(f"{prefix} {key}: {self._color(str(value), Colors.YELLOW)}")

        print(f"\n{separator}\n")

    def print_tuning_start(self):
        """Print message when tuning starts."""
        print(
            self._format_message(
                "🏁", "Starting hyperparameter optimization...", Colors.GREEN
            )
        )

    def print_tuning_status(self, tuning_status: TuningStatus):
        """Print the current tuning status."""
        separator = "━" * 80
        print(f"\n{separator}")
        tuning_status_msg = "Tuning Status"
        if self.use_emojis:
            tuning_status_msg = "📊 " + tuning_status_msg
        print(
            f"{self._color(tuning_status_msg, Colors.BOLD)} (last metric is reported)"
        )
        print(str(tuning_status))
        print(f"{separator}\n")

    def print_config_space_exhausted(self):
        """Print message when configuration space is exhausted."""
        print(
            self._format_message(
                "🎊",
                "Configuration space exhausted! Waiting for running trials to complete...",
                Colors.CYAN,
            )
        )
        print("Tuning is finishing as the whole configuration space got exhausted.")

    def print_trial_started(self, trial_id: int, config: dict):
        """Print message when a trial is started."""
        timestamp = self._get_timestamp()
        # Format config more compactly
        config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
        print(
            f"{timestamp} {self._format_message('🚀', f'Trial {trial_id} started - config: {config_str}', Colors.GREEN)}"
        )

    def print_trial_resumed(self, trial_id: int, config: dict | None = None):
        """Print message when a trial is resumed."""
        timestamp = self._get_timestamp()
        log_msg = f"Resuming trial {trial_id}"
        if config is not None:
            log_msg += f" with new config: {config}"
        print(f"{timestamp} {self._format_message('▶️', log_msg, Colors.CYAN)}")

    def print_trial_completed(self, trial_id: int):
        """Print message when a trial completes."""
        timestamp = self._get_timestamp()
        print(
            f"{timestamp} {self._format_message('✅', f'Trial {trial_id} completed!', Colors.GREEN)}"
        )

    def print_trial_result(
        self, trial_id: int, result: dict, epoch: int = None, total_epochs: int = None
    ):
        """Print intermediate result from a trial."""
        timestamp = self._get_timestamp()
        # Format result metrics
        metrics_str = " | ".join(
            [
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in result.items()
                if k not in ["trial_id", "epoch", "resource_attr"]
            ]
        )

        epoch_str = ""
        if epoch is not None and total_epochs is not None:
            epoch_str = f"Epoch {epoch}/{total_epochs} | "
        elif epoch is not None:
            epoch_str = f"Epoch {epoch} | "

        print(
            f"{timestamp} {self._format_message('📊', f'Trial {trial_id} | {epoch_str}{metrics_str}', Colors.BLUE)}"
        )

    def print_trial_failed(self, trial_id: int):
        """Print message when a trial fails."""
        timestamp = self._get_timestamp()
        print(
            f"{timestamp} {self._format_message('❌', f'Trial {trial_id} failed.', Colors.RED)}"
        )

    def print_trial_stopped_by_scheduler(self, trial_id: int):
        """Print message when a trial is stopped by scheduler."""
        timestamp = self._get_timestamp()
        print(
            f"{timestamp} {self._format_message('🛑', f'Trial {trial_id} stopped by scheduler.', Colors.YELLOW)}"
        )

    def print_trial_stopped_independently(self, trial_id: int):
        """Print message when a trial is stopped independently."""
        timestamp = self._get_timestamp()
        print(
            f"{timestamp} {self._format_message('🛑', f'Trial {trial_id} was stopped independently of the scheduler.', Colors.YELLOW)}"
        )

    def print_trial_paused(self, trial_id: int):
        """Print message when a trial is paused."""
        timestamp = self._get_timestamp()
        print(
            f"{timestamp} {self._format_message('⏸️', f'Trial {trial_id} paused by scheduler.', Colors.YELLOW)}"
        )

    def print_no_metrics_observed(self, trial_id: int, stdout: str, stderr: str):
        """Print error when trial completes without metrics."""
        timestamp = self._get_timestamp()
        print(
            f"{timestamp} {self._format_message('❌', f'Trial {trial_id} completed but no metrics were observed. Check logs:', Colors.RED)}"
        )
        print(f"\n{self._color('STDOUT:', Colors.RED)}")
        print(stdout)
        print(f"\n{self._color('STDERR:', Colors.RED)}")
        print(stderr)

    def print_searcher_out_of_candidates(self):
        """Print message when searcher runs out of candidates."""
        print(
            self._format_message(
                "🔍",
                "Searcher ran out of candidates, tuning job is stopping.",
                Colors.CYAN,
            )
        )

    def print_tuning_complete(self):
        """Print completion banner."""
        separator = "━" * 80
        print(f"\n{separator}")
        print(
            self._format_message(
                "🎉🎉🎉",
                "HYPERPARAMETER OPTIMIZATION COMPLETE!",
                Colors.GREEN + Colors.BOLD,
            )
        )
        print(f"{separator}\n")

    def print_stopping_trials(self):
        """Print message when stopping remaining trials."""
        print(
            self._format_message(
                "🛑", "Stopping trials that may still be running.", Colors.YELLOW
            )
        )

    def print_tuning_finished(self, results_path: Path):
        """Print final message with results location."""
        print(
            self._format_message(
                "✅",
                f"Tuning finished, results of trials can be found at {results_path}",
                Colors.GREEN,
            )
        )
        print(
            f"\n💾 {self._color('Results saved to:', Colors.BOLD)} {self._color(str(results_path), Colors.BLUE)}"
        )
        print(
            f"✨ {self._color('Happy training with your optimized hyperparameters!', Colors.CYAN)}\n"
        )

    def print_error(self, message: str):
        """Print error message."""
        print(self._format_message("💥", message, Colors.RED))

    def print_warning(self, message: str):
        """Print warning message."""
        print(self._format_message("⚠️", message, Colors.YELLOW))

    def print_max_failures_reached(self, max_failures: int):
        """Print message when max failures are reached."""
        print(
            self._format_message(
                "❌", f"Stopped as {max_failures} failures were reached", Colors.RED
            )
        )

    def print_failure_logs(self, trial_id: int, stdout: str, stderr: str):
        """Print logs from a failed trial."""
        print(
            self._format_message(
                "📋", f"Showing log of first failure (Trial {trial_id})", Colors.RED
            )
        )
        print(f"\n{self._color('STDOUT:', Colors.RED)}")
        print(stdout)
        print(f"\n{self._color('STDERR:', Colors.RED)}")
        print(stderr)

    def print_best_config_instructions(self, trial_id: int, config: dict):
        """Print instructions for retraining with best config."""
        print(
            f"\n{self._color('💡 To retrain with the best configuration:', Colors.CYAN + Colors.BOLD)}\n"
        )
        print(f"   {self._color('Start from scratch:', Colors.CYAN)}")
        print(f"   >>> tuner.trial_backend.start_trial(config={config})\n")
        print(f"   {self._color('Resume from checkpoint:', Colors.CYAN)}")
        print(
            f"   >>> tuner.trial_backend.start_trial(config={config}, checkpoint_trial_id={trial_id})\n"
        )

    def print_scheduler_deprecated(self, scheduler_name: str):
        """Print deprecation warning for scheduler."""
        print(
            self._format_message(
                "⚠️",
                f"Scheduler {scheduler_name} is deprecated and will be removed in the next release!",
                Colors.YELLOW,
            )
        )
