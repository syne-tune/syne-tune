from pathlib import Path
import logging
from typing import Dict, Optional, List
import copy
import pytest

from sagemaker.pytorch import PyTorch

from syne_tune.backend import SageMakerBackend
from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
from syne_tune.optimizer.scheduler import TrialScheduler, TrialSuggestion, \
    Trial
from syne_tune.config_space import randint
from syne_tune import StoppingCriterion, Tuner

logger = logging.getLogger(__name__)


class TestCopyCheckpointScheduler(TrialScheduler):
    """
    Scheduler for `test_copy_checkpoint_sagemaker_backend`.

    """
    def __init__(self, config_space: Dict):
        super().__init__(config_space)
        self.result_for_trial = dict()

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        config = {
            'trial_id': trial_id,
            'value1': trial_id + 1,
            'value2': 2 * (trial_id + 1),
        }
        if trial_id == 1:
            # Start new trial from checkpoint of trial_id 0 (this requires
            # copying the checkpoint)
            return TrialSuggestion.start_suggestion(
                config=config, checkpoint_trial_id=0)
        else:
            # Start new trial from scratch
            return TrialSuggestion.start_suggestion(config=config)

    def on_trial_complete(self, trial: Trial, result: Dict):
        trial_id = trial.trial_id
        logger.info(
            f"on_trial_complete (trial_id = {trial_id}). Received:\n{result}")
        self.result_for_trial[trial_id] = copy.copy(result)

    def metric_names(self) -> List[str]:
        return ['value1']


@pytest.mark.skip("this test needs sagemaker and runs for >10 minutes")
def test_copy_checkpoint_sagemaker_backend():
    logging.getLogger().setLevel(logging.INFO)
    # Create SageMaker backend
    entry_point = Path(__file__).parent / "checkpoint_script.py"
    trial_backend = SageMakerBackend(
        sm_estimator=PyTorch(
            entry_point=str(entry_point),
            instance_type="ml.m5.large",
            instance_count=1,
            role=get_execution_role(),
            max_run=10 * 60,
            framework_version='1.7.1',
            py_version='py3',
        ))

    config_space = {
        'trial_id': randint(0, 9),
        'value1': randint(0, 20),
        'value2': randint(0, 20),
    }
    test_scheduler = TestCopyCheckpointScheduler(config_space)

    stop_criterion = StoppingCriterion(max_num_trials_completed=1)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=test_scheduler,
        stop_criterion=stop_criterion,
        n_workers=1,
        tuner_name='test-copy-checkpoint',
        callbacks=[],
    )

    logger.info(f"Starting tuning: {tuner.name}")
    tuner.run()
    logger.info("Done tuning. Checking results")

    result_for_trial = test_scheduler.result_for_trial
    logger.info(f"result_for_trial:\n{result_for_trial}")
    # Normal, no checkpoint copied from parent
    assert 0 in result_for_trial
    result0 = result_for_trial[0]
    assert 'error_msg' not in result0, result0['error_msg']
    assert result0['trial_id'] == 0
    assert result0['value1'] == 1
    assert result0['value2'] == 2
    assert 'parent1_trial_id' not in result0
    assert 'parent2_trial_id' not in result0
    # Checkpoint copied from parent trial_id=0
    assert 1 in result_for_trial
    result1 = result_for_trial[1]
    assert 'error_msg' not in result1, result1['error_msg']
    assert result1['trial_id'] == 1
    assert result1['value1'] == 2
    assert result1['value2'] == 4
    assert result1['parent1_trial_id'] == 0
    assert result1['parent1_value1'] == 1
    assert result1['parent2_trial_id'] == 0
    assert result1['parent2_value2'] == 2
