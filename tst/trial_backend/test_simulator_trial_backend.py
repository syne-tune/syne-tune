from typing import Dict, Any
import math
import pytest

from syne_tune.backend import LocalBackend
from syne_tune.backend.simulator_backend.simulator_backend import (
    SimulatorBackend,
    SimulatorConfig,
)
from syne_tune.backend.simulator_backend.events import (
    SimulatorState,
    StartEvent,
    CompleteEvent,
    OnTrialResultEvent,
)
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.results_callback import StoreResultsCallback
from syne_tune import StoppingCriterion
from syne_tune import Tuner
from syne_tune.constants import ST_DECISION, ST_TRIAL_ID
from syne_tune.optimizer.schedulers.legacy_hyperband import LegacyHyperbandScheduler
from syne_tune.optimizer.schedulers.legacy_fifo import LegacyFIFOScheduler
from syne_tune.optimizer.scheduler import SchedulerDecision


def _compare_results(res_local: Dict[str, Any], res_simul: Dict[str, Any], num: int):
    for key in (ST_TRIAL_ID, ST_DECISION, "epoch", "mean_loss"):
        rloc = res_local[key]
        rsim = res_simul[key]
        if key != "mean_loss":
            assert_cond = rloc == rsim
        else:
            assert_cond = math.isclose(rloc, rsim, rel_tol=1e-6)
        assert assert_cond, (
            f"{num}, {key}: local = {rloc} != {rsim} = simul\n"
            + f"res_local = {res_local}\n"
            + f"res_simul = {res_simul}"
        )


# Note: This test is very tricky to get right. When training times are very
# short, trivial differences between local and simulated backend get
# amplified. These do not play a role with realistic training times of
# more than 1 sec per epoch.
@pytest.mark.skip(
    "skipping for now since it depends on examples which is not included in path"
)
@pytest.mark.parametrize("scheduler_name", ["fifo"])
def test_compare_local_simulator_backends(scheduler_name):
    from examples.training_scripts.height_with_cost.train_height_with_cost import (
        height_with_cost_default_params,
        height_with_cost_benchmark,
    )

    random_seed = 382378624
    n_workers = 4
    tuner_sleep_time = 0.1
    # For 'bayesopt', fixing the seed does not render an experiment entirely
    # deterministic
    searcher_name = "random"

    default_params = height_with_cost_default_params()
    default_params["max_resource_level"] = 9  # To make it run faster
    benchmark = height_with_cost_benchmark(default_params)
    # Benchmark must be tabulated to support simulation:
    assert benchmark.get("supports_simulated", False)

    if scheduler_name == "fifo":
        stop_criterion = StoppingCriterion(max_num_trials_completed=6)
    else:
        stop_criterion = StoppingCriterion(max_num_trials_started=15)
    # Run experiment with two different backends
    results = dict()
    for backend_name in ("local", "simulated"):
        benchmark["config_space"]["dont_sleep"] = backend_name == "simulated"
        # Create scheduler
        # search_options = {'debug_log': False}
        search_options = {"debug_log": True}
        scheduler_options = {
            "searcher": searcher_name,
            "search_options": search_options,
            "metric": benchmark["metric"],
            "mode": benchmark["mode"],
            "random_seed": random_seed,
        }
        if scheduler_name != "fifo":
            sch_type = scheduler_name[len("hyperband_") :]
            scheduler_options.update(
                {
                    "resource_attr": benchmark["resource_attr"],
                    "type": sch_type,
                    "grace_period": 1,
                    "reduction_factor": 3,
                }
            )
        scheduler_cls = (
            LegacyFIFOScheduler
            if scheduler_name == "fifo"
            else LegacyHyperbandScheduler
        )
        scheduler = scheduler_cls(benchmark["config_space"], **scheduler_options)
        # Create backend
        if backend_name == "local":
            trial_backend = LocalBackend(entry_point=benchmark["script"])
        else:
            simulator_config = SimulatorConfig(
                delay_on_trial_result=0,
                delay_complete_after_final_report=0,
                delay_complete_after_stop=0,
                delay_start=0,
                delay_stop=0,
            )
            trial_backend = SimulatorBackend(
                entry_point=benchmark["script"],
                elapsed_time_attr=benchmark["elapsed_time_attr"],
                simulator_config=simulator_config,
                tuner_sleep_time=tuner_sleep_time,
            )
            scheduler.set_time_keeper(trial_backend.time_keeper)

        _tuner_sleep_time = 0 if backend_name == "simulated" else tuner_sleep_time
        # Run experiment
        if backend_name == "local":
            # Duplicates callback used in ``tuner.run``, but we have access
            result_callback = StoreResultsCallback()
        else:
            result_callback = SimulatorCallback()
        local_tuner = Tuner(
            trial_backend=trial_backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=n_workers,
            sleep_time=_tuner_sleep_time,
            results_update_interval=100,
            print_update_interval=100,
            callbacks=[result_callback],
        )
        local_tuner.run()
        results[backend_name] = result_callback.results

    # Compare results. Note that times are not comparable. We may not see
    # exactly the same number of results, but the prefix of results should
    # be the same.
    # Note: Differences between the two are mainly due to very short training
    # times, which amplify the importance of processing in ``LocalBackend`` we
    # do not simulate (e.g., starting subprocesses).
    # Filter results for stopped trials which are received after the stop
    # rung. These are filtered out by the simulator backend, but not by
    # the local backend
    rung_levels = (9, 3, 1)
    max_resource_when_stopped = dict()
    stop_decisions = {SchedulerDecision.STOP, SchedulerDecision.PAUSE}
    for result in results["local"]:
        decision = result[ST_DECISION]
        if decision in stop_decisions:
            trial_id = result[ST_TRIAL_ID]
            resource = int(result["epoch"])
            max_resource_when_stopped[trial_id] = resource
    for trial_id, max_resource in max_resource_when_stopped.items():
        for rung_level in rung_levels:
            if max_resource >= rung_level:
                max_resource_when_stopped[trial_id] = rung_level
                break
    new_local = []
    for result in results["local"]:
        trial_id = result[ST_TRIAL_ID]
        resource = int(result["epoch"])
        decision = result[ST_DECISION]
        if (
            decision not in stop_decisions
            or resource <= max_resource_when_stopped[trial_id]
        ):
            new_local.append(result)
    results["local"] = new_local
    num_local = len(results["local"])
    num_simul = len(results["simulated"])
    if num_local != num_simul:
        print(
            f"{scheduler_name}: num_results_local = {num_local}, num_results_simul = {num_simul}"
        )

    def sort_key(result):
        return (result[ST_TRIAL_ID], result["epoch"])

    if num_local <= num_simul:
        k_short = "local"
        k_long = "simulated"
    else:
        k_long = "local"
        k_short = "simulated"
    dict_long = {sort_key(v): v for v in results[k_long]}
    new_short, new_long = [], []
    for v in results[k_short]:
        k = sort_key(v)
        if k in dict_long:
            new_long.append(dict_long[k])
            new_short.append(v)
    results[k_short] = new_short
    results[k_long] = new_long
    assert len(new_short) >= 0.7 * max(
        num_local, num_simul
    ), f"num_matched = {len(new_short)}, num_results_local = {num_local}, num_results_simul = {num_simul}"
    for i, (res_local, res_simul) in enumerate(
        zip(results["local"], results["simulated"])
    ):
        _compare_results(res_local, res_simul, i)


def test_simulator_state():
    state = SimulatorState()
    result1 = dict(epoch=1, accuracy=0.5)
    result2 = dict(epoch=2, accuracy=0.75)
    result3 = dict(epoch=3, accuracy=0.8)
    state.push(StartEvent(trial_id=0), event_time=1)
    state.push(OnTrialResultEvent(trial_id=0, result=result1), event_time=2)
    state.push(OnTrialResultEvent(trial_id=0, result=result2), event_time=2)
    state.push(OnTrialResultEvent(trial_id=0, result=result3), event_time=3)
    state.push(CompleteEvent(trial_id=0, status="completed"), event_time=3.2)
    state.push(StartEvent(trial_id=1), event_time=1)
    state.push(OnTrialResultEvent(trial_id=1, result=result1), event_time=2.5)
    state.push(OnTrialResultEvent(trial_id=1, result=result2), event_time=2.6)
    state.push(OnTrialResultEvent(trial_id=1, result=result3), event_time=3.5)
    state.push(CompleteEvent(trial_id=1, status="completed"), event_time=4)
    # Everything until 2.2
    required_results = [
        (StartEvent, 0, 1.0),
        (StartEvent, 1, 1.0),
        (OnTrialResultEvent, 0, 1, 2.0),
        (OnTrialResultEvent, 0, 2, 2.0),
    ]
    for i in range(2):
        obtained_results = []
        time_until = 2.2 if i == 0 else 4
        while True:
            entry = state.next_until(time_until)
            if entry is None:
                break
            else:
                obtained_results.append(entry)
        assert len(required_results) == len(
            obtained_results
        ), f"i={i}: {required_results}\n{obtained_results}"
        for j, (res_req, res_obt) in enumerate(zip(required_results, obtained_results)):
            assert res_req[-1] == res_obt[0], (i, j, res_req[-1], res_obt[0])
            event = res_obt[1]
            assert isinstance(event, res_req[0])
            assert event.trial_id == res_req[1]
            if isinstance(event, OnTrialResultEvent):
                assert event.result["epoch"] == res_req[2]
            elif isinstance(event, CompleteEvent):
                assert event.status == res_req[2]
        if i == 0:
            state.remove_events(trial_id=0)
        required_results = [
            (OnTrialResultEvent, 1, 1, 2.5),
            (OnTrialResultEvent, 1, 2, 2.6),
            (OnTrialResultEvent, 1, 3, 3.5),
            (CompleteEvent, 1, "completed", 4),
        ]
