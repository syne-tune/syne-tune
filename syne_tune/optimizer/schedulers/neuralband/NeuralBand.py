# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import copy
import logging
import os
from typing import Dict, Optional, List

import numpy as np

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import SchedulerDecision
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.hyperband_cost_promotion import (
    CostPromotionRungSystem,
)
from syne_tune.optimizer.schedulers.hyperband_pasha import PASHARungSystem
from syne_tune.optimizer.schedulers.hyperband_promotion import PromotionRungSystem
from syne_tune.optimizer.schedulers.hyperband_rush import (
    RUSHPromotionRungSystem,
    RUSHStoppingRungSystem,
)
from syne_tune.optimizer.schedulers.hyperband_stopping import StoppingRungSystem
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    check_and_merge_defaults,
    Integer,
    Boolean,
    Categorical,
    filter_by_key,
    String,
    Dictionary,
    Float,
)

from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher
from syne_tune.optimizer.schedulers.searchers.searcher_factory import searcher_factory


logger = logging.getLogger(__name__)

def is_continue_decision(trial_decision: str) -> bool:
    return trial_decision == SchedulerDecision.CONTINUE


from syne_tune.optimizer.schedulers.hyperband import HyperbandBracketManager, _ARGUMENT_KEYS, _CONSTRAINTS
from syne_tune.optimizer.scheduler import (
    TrialScheduler,
    SchedulerDecision,
    TrialSuggestion,
)

from syne_tune.optimizer.schedulers.hyperband import _get_rung_levels, _is_positive_int, _sample_bracket

from syne_tune.config_space import cast_config_values

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges import (
    HyperparameterRanges,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory import (
     make_hyperparameter_ranges,
)

from syne_tune.optimizer.schedulers.neuralband.networks import Exploitation



_DEFAULT_OPTIONS = {
    "resource_attr": "epoch",
    "resume": False,
    "grace_period": 1,
    "reduction_factor": 3,
    "brackets": 1,
    "type": "stopping",
    "searcher_data": "rungs",
    "register_pending_myopic": False,
    "do_snapshots": False,
    "rung_system_per_bracket": False,
    "rung_system_kwargs": {
        "ranking_criterion": "soft_ranking",
        "epsilon": 1.0,
        "epsilon_scaling": 1.0,
    },
}



class NeuralbandScheduler(FIFOScheduler):

    def __init__(self, config_space, gamma = 0.01, nu = 0.01, step_size = 30, max_while_loop = 100,  **kwargs):
        # Before we can call the superclass constructor, we need to set a few
        # members (see also `_extend_search_options`).
        # To do this properly, we first check values and impute defaults for
        # `kwargs`.
        
        kwargs = check_and_merge_defaults(
            kwargs, set(), _DEFAULT_OPTIONS, _CONSTRAINTS, dict_name="scheduler_options"
        )
        scheduler_type = kwargs["type"]
        self.scheduler_type = scheduler_type
        self._resource_attr = kwargs["resource_attr"]
        self._rung_system_kwargs = kwargs["rung_system_kwargs"]
        self._cost_attr = kwargs.get("cost_attr")
        assert not (
            scheduler_type == "cost_promotion" and self._cost_attr is None
        ), "cost_attr must be given if type='cost_promotion'"
        # Superclass constructor
        resume = kwargs["resume"]
        kwargs["resume"] = False  # Cannot be done in superclass
        super().__init__(config_space, **filter_by_key(kwargs, _ARGUMENT_KEYS))
        assert self.max_t is not None, (
            "Either max_t must be specified, or it has to be specified as "
            + "config_space['epochs'], config_space['max_t'], "
            + "config_space['max_epochs']"
        )

        # If rung_levels is given, grace_period and reduction_factor are ignored
        rung_levels = kwargs.get("rung_levels")
        
        if rung_levels is not None:
            assert isinstance(rung_levels, list)
            if ("grace_period" in kwargs) or ("reduction_factor" in kwargs):
                logger.warning(
                    "Since rung_levels is given, the values grace_period = "
                    f"{kwargs.get('grace_period')} and reduction_factor = "
                    f"{kwargs.get('reduction_factor')} are ignored!"
                )
        rung_levels = _get_rung_levels(
            rung_levels,
            grace_period=kwargs["grace_period"],
            reduction_factor=kwargs["reduction_factor"],
            max_t=self.max_t,
        )
        
        brackets = kwargs["brackets"]
        do_snapshots = kwargs["do_snapshots"]
        assert (not do_snapshots) or (
            scheduler_type == "stopping"
        ), "Snapshots are supported only for type = 'stopping'"
        rung_system_per_bracket = kwargs["rung_system_per_bracket"]

        self.terminator = HyperbandBracketManager(
            scheduler_type,
            self._resource_attr,
            self.metric,
            self.mode,
            self.max_t,
            rung_levels,
            brackets,
            rung_system_per_bracket,
            cost_attr=self._total_cost_attr(),
            random_seed=self.random_seed_generator(),
            rung_system_kwargs=self._rung_system_kwargs,
        )
        self.do_snapshots = do_snapshots
        self.searcher_data = kwargs["searcher_data"]
        self._register_pending_myopic = kwargs["register_pending_myopic"]
        self._active_trials = dict()
        
        
        
        self.kwargs = kwargs
        
        
        print("reduction_factor", kwargs["reduction_factor"])
        
        # added code ---------- ban -------------  
        # encoding
        self.hp_ranges = make_hyperparameter_ranges(config_space = self.config_space)
        self.input_dim = self.hp_ranges.ndarray_size
           
        #neural network
        self.net = Exploitation(dim = self.input_dim)
        self.net.brackets = kwargs["brackets"]
        
        self.currnet_best_score = 1.0 
        self.gamma = gamma
        self.nu = nu
        if self.mode == "min":
            self.train_step_size = step_size
        else:
            self.train_step_size = 2
                
        # how many trails we train network once
        self.max_while_loop = max_while_loop
        # ------------- ban ----------------
        
        
        self._cost_offset = dict()
        if resume:
            checkpoint = kwargs.get("checkpoint")
            assert checkpoint is not None, "Need checkpoint to be set if resume = True"
            if os.path.isfile(checkpoint):
                raise NotImplementedError()
                # TODO! Need load
                # self.load_state_dict(load(checkpoint))
            else:
                msg = f"checkpoint path {checkpoint} is not available for resume."
                logger.exception(msg)
                raise FileExistsError(msg)


            
    def _initialize_searcher_new(self): 
            searcher = self.kwargs["searcher"]     
            print("configurations run out and initialize the searcher", searcher)
            search_options = self.kwargs.get("search_options")
            if search_options is None:
                search_options = dict()
            else:
                search_options = search_options.copy()
            search_options.update(
                {
                    "config_space": self.config_space.copy(),
                    "metric": self.metric,
                    "points_to_evaluate": self.kwargs.get("points_to_evaluate"),
                    "scheduler_mode": self.kwargs["mode"],
                    "mode": self.kwargs["mode"],
                    "random_seed_generator": self.random_seed_generator,
                }
            )
            if self.max_t is not None:
                search_options["max_epochs"] = self.max_t
            # Subclasses may extend `search_options`
            search_options = self._extend_search_options(search_options)
            # Adjoin scheduler info to search_options, if not already done by
            # subclass (via `_extend_search_options`)
            if "scheduler" not in search_options:
                search_options["scheduler"] = "fifo"
            self.searcher: BaseSearcher = searcher_factory(searcher, **search_options)
            self._searcher_initialized = True

            
    
    
        
    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
                
        self._initialize_searcher()
        # If no time keeper was provided at construction, we use a local
        # one which is started here
        if self.time_keeper is None:
            self.time_keeper = RealTimeKeeper()
            self.time_keeper.start_of_time()
        # For pause/resume schedulers: Can a paused trial be promoted?
        promote_trial_id, extra_kwargs = self._promote_trial()
        if promote_trial_id is not None:
            promote_trial_id = int(promote_trial_id)
            return TrialSuggestion.resume_suggestion(
                trial_id=promote_trial_id, config=extra_kwargs
            )
        # Ask searcher for config of new trial to start
        extra_kwargs["elapsed_time"] = self._elapsed_time()
        trial_id = str(trial_id)
        
        # added code ----------- Ban ------------
        initial_budget = self.net.max_b
        while_loop_count = 0
        l_t_score = []
        while 1:
            config = self.searcher.get_config(**extra_kwargs, trial_id=trial_id)
           
            
            if config is not None:
                config_encoding =  self.hp_ranges.to_ndarray(config)
                predict_score = self.net.predict((config_encoding, initial_budget)).item()
                l_t_score.append((config, predict_score))
               
                
                
                if self.mode == "min":

                    if self.currnet_best_score - predict_score > self.gamma * self.currnet_best_score * (1.0 - initial_budget / self.max_t):
                        break
                    
                    if while_loop_count > self.max_while_loop:
                        l_t_score = sorted(l_t_score, key = lambda x: x[1])
                        config = l_t_score[0][0]
                        break
                        
                else:
                    if predict_score*100.0 - self.currnet_best_score > self.gamma * (100.0 - self.currnet_best_score ) * (1.0 - initial_budget / self.max_t):
                        break
                    
                    if while_loop_count > (self.max_while_loop/20):
                        l_t_score = sorted(l_t_score, key = lambda x: x[1], reverse=True)
                        config = l_t_score[0][0]
                        break
                while_loop_count += 1      
                
                    
            else:
                self._searcher_initialized = False
                self._initialize_searcher_new()
                config = self.searcher.get_config(**extra_kwargs, trial_id=trial_id)
                break
                
        #------------- Ban ----------------
        #print(predict_score)   
        if config is not None:
            config = cast_config_values(config, self.config_space)
            config = self._on_config_suggest(config, trial_id, **extra_kwargs)
            config = TrialSuggestion.start_suggestion(config)
            
        #pdb.set_trace()   
        return config    

        

    def _on_config_suggest(self, config: Dict, trial_id: str, **kwargs) -> Dict:
        """
        `kwargs` being used here:
        - elapsed_time: Time from start of experiment, set in
            `FIFOScheduler._suggest`
        - bracket: Bracket in which new trial is started, set in
            `HyperbandScheduler._promote_trial`
        - milestone: First milestone the new trial will reach, set in
            `HyperbandScheduler._promote_trial`
        """
        assert trial_id not in self._active_trials, f"Trial {trial_id} already exists"
        # See `FIFOScheduler._on_config_suggest` for why we register the task
        # and pending evaluation here, and not later in `on_task_add`.
        debug_log = self.searcher.debug_log
                
        # Register new task
        first_milestone = self.terminator.on_task_add(
            trial_id, bracket=kwargs["bracket"], new_config=True
        )[-1]
        if debug_log is not None:
            logger.info(
                f"trial_id {trial_id} starts (first milestone = " f"{first_milestone})"
            )
        # Register pending evaluation with searcher
        if self.searcher_data == "rungs":
            pending_resources = [first_milestone]
        elif self._register_pending_myopic:
            pending_resources = [1]
        else:
            pending_resources = list(range(1, first_milestone + 1))
        for resource in pending_resources:
            self.searcher.register_pending(
                trial_id=trial_id, config=config, milestone=resource
            )
            
       
        # Extra fields in `config`
        if debug_log is not None:
            # For log outputs:
            config["trial_id"] = trial_id
        if self.does_pause_resume() and self.max_resource_attr is not None:
            # The new trial should only run until the next milestone.
            # This needs its config to be modified accordingly.
            config[self.max_resource_attr] = kwargs["milestone"]

        self._active_trials[trial_id] = {
            "config": copy.copy(config),
            "time_stamp": kwargs["elapsed_time"],
            "bracket": kwargs["bracket"],
            "reported_result": None,
            "keep_case": False,
            "running": True,
            "largest_update_resource": None,
        }
        return config


    def _promote_trial(self) -> (Optional[str], Optional[Dict]):
        #print("promote trial")

        trial_id, extra_kwargs = self.terminator.on_task_schedule()
        if trial_id is None:
            # No trial to be promoted
            if self.do_snapshots:
                # Append snapshot
                bracket_id = extra_kwargs["bracket"]
                extra_kwargs["snapshot"] = {
                    "tasks": self._snapshot_tasks(bracket_id),
                    "rungs": self.terminator.snapshot_rungs(bracket_id),
                    "max_resource": self.max_t,
                }
        else:
            # At this point, we can assume the trial will be resumed
            extra_kwargs["new_config"] = False
            self.terminator.on_task_add(trial_id, **extra_kwargs)
            # Update information (note that 'time_stamp' is not exactly
            # correct, since the task may get started a little later)
            assert (
                trial_id in self._active_trials
            ), f"Paused trial {trial_id} must be in _active_trials"
            record = self._active_trials[trial_id]
            assert not record[
                "running"
            ], f"Paused trial {trial_id} marked as running in _active_trials"
            record.update(
                {
                    "time_stamp": self._elapsed_time(),
                    "reported_result": None,
                    "keep_case": False,
                    "running": True,
                }
            )
            # Register pending evaluation(s) with searcher
            next_milestone = extra_kwargs["milestone"]
            resume_from = extra_kwargs["resume_from"]
            if self.searcher_data == "rungs":
                pending_resources = [next_milestone]
            elif self._register_pending_myopic:
                pending_resources = [resume_from + 1]
            else:
                pending_resources = list(range(resume_from + 1, next_milestone + 1))
            for resource in pending_resources:
                self.searcher.register_pending(trial_id=trial_id, milestone=resource)
            if self.searcher.debug_log is not None:
                logger.info(
                    f"trial_id {trial_id}: Promotion from "
                    f"{resume_from} to {next_milestone}"
                )
            # In the case of a promoted trial, extra_kwargs plays a different
            # role
            if self.does_pause_resume() and self.max_resource_attr is not None:
                # The promoted trial should only run until the next milestone.
                # This needs its config to be modified accordingly
                extra_kwargs = record["config"].copy()
                extra_kwargs[self.max_resource_attr] = next_milestone
            else:
                extra_kwargs = None
        return trial_id, extra_kwargs

    def _snapshot_tasks(self, bracket_id):
        # If all brackets share a single rung level system, then all
        # running jobs have to be taken into account, otherwise only
        # those jobs running in the same bracket
        all_running = not self.terminator._rung_system_per_bracket
        tasks = dict()
        for k, v in self._active_trials.items():
            if v["running"] and (all_running or v["bracket"] == bracket_id):
                reported_result = v["reported_result"]
                level = (
                    0
                    if reported_result is None
                    else reported_result[self._resource_attr]
                )
                # It is possible to have tasks in _active_trials which have
                # reached self.max_t. These must not end up in the snapshot
                if level < self.max_t:
                    tasks[k] = {
                        "config": v["config"],
                        "time": v["time_stamp"],
                        "level": level,
                    }
        return tasks

    def _cleanup_trial(self, trial_id: str):
        """
        Called for trials which are stopped or paused. The trial is still kept
        in the records.
        :param trial_id:
        """
        self.terminator.on_task_remove(trial_id)
        if trial_id in self._active_trials:
            # We do not remove stopped trials
            self._active_trials[trial_id]["running"] = False

    def on_trial_error(self, trial: Trial):
        super().on_trial_error(trial)
        self._cleanup_trial(str(trial.trial_id))

    def _update_searcher_internal(self, trial_id: str, config: Dict, result: Dict):
        if self.searcher_data == "rungs_and_last":
            # Remove last recently added result for this task. This is not
            # done if it fell on a rung level (i.e., `keep_case` is True)
            record = self._active_trials[trial_id]
            if (record["reported_result"] is not None) and (not record["keep_case"]):
                rem_result = record["reported_result"]
                self.searcher.remove_case(trial_id, **rem_result)

    def _update_searcher(
        self, trial_id: str, config: Dict, result: Dict, task_info: Dict
    ):
        """
        Updates searcher with `result` (depending on `searcher_data`), and
        registers pending config with searcher.
        :param trial_id:
        :param config:
        :param result: Record obtained from `on_trial_result`
        :param task_info: Info from `self.terminator.on_task_report`
        :return: Should searcher be updated?
        """
        task_continues = task_info["task_continues"]
        milestone_reached = task_info["milestone_reached"]
        next_milestone = task_info.get("next_milestone")
        do_update = False
        pending_resources = []
        if self.searcher_data == "rungs":
            if milestone_reached:
                # Update searcher with intermediate result
                do_update = True
                if task_continues and next_milestone is not None:
                    pending_resources = [next_milestone]
        elif not task_info.get("ignore_data", False):
            do_update = True
            if task_continues:
                resource = int(result[self._resource_attr])
                if self._register_pending_myopic or next_milestone is None:
                    pending_resources = [resource + 1]
                elif milestone_reached:
                    # Register pending evaluations for all resources up to
                    # `next_milestone`
                    pending_resources = list(range(resource + 1, next_milestone + 1))
        # Update searcher
        if do_update:
            self._update_searcher_internal(trial_id, config, result)
        # Register pending evaluations
        for resource in pending_resources:
            self.searcher.register_pending(
                trial_id=trial_id, config=config, milestone=resource
            )
        return do_update

    def _check_result(self, result: Dict):
        #print("check result")
        super()._check_result(result)
        self._check_key_of_result(result, self._resource_attr)
        if self.scheduler_type == "cost_promotion":
            self._check_key_of_result(result, self._cost_attr)
        resource = result[self._resource_attr]
        assert 1 <= resource == round(resource), (
            "Your training evaluation function needs to report positive "
            + f"integer values for key {self._resource_attr}. Obtained "
            + f"value {resource}, which is not permitted"
        )

    def on_trial_result(self, trial: Trial, result: Dict) -> str:
        self._check_result(result)
        trial_id = str(trial.trial_id)
        debug_log = self.searcher.debug_log
        trial_decision = SchedulerDecision.CONTINUE
        if len(result) == 0:
            # An empty dict should just be skipped
            if debug_log is not None:
                logger.info(
                    f"trial_id {trial_id}: Skipping empty dict received "
                    "from reporter"
                )
        else:
            
            # added code
            # Time since start of experiment
            time_since_start = self._elapsed_time()
            do_update = False
            config = self._preprocess_config(trial.config)
            cost_and_promotion = (
                self._cost_attr is not None
                and self._cost_attr in result
                and self.does_pause_resume()
            )
            if cost_and_promotion:
                # Trial may have paused/resumed before, so need to add cost
                # offset from these
                cost_offset = self._cost_offset.get(trial_id, 0)
                result[self._total_cost_attr()] = result[self._cost_attr] + cost_offset
            if trial_id not in self._active_trials:
                # Trial not in self._active_trials anymore, so must have been
                # stopped
                trial_decision = SchedulerDecision.STOP
                logger.warning(
                    f"trial_id {trial_id}: Was STOPPED, but receives another "
                    f"report {result}\nThis report is ignored"
                )
            elif not self._active_trials[trial_id]["running"]:
                # Trial must have been paused before
                trial_decision = SchedulerDecision.PAUSE
                logger.warning(
                    f"trial_id {trial_id}: Was PAUSED, but receives another "
                    f"report {result}\nThis report is ignored"
                )
            else:
                
                
                
                # added code -------------- Ban ------------------
                
                config = trial.config
                config_encoding =  self.hp_ranges.to_ndarray(config)
                if "epoch" in result:
                    hp_budget = float(result["epoch"]/self.max_t) 
                else:    
                    hp_budget = float(result["hp_epoch"]/self.max_t) 
                
                
                test_loss = result[self.metric]
                
                # update current best score
                if self.mode == "min":
                    if test_loss < self.currnet_best_score:
                        self.currnet_best_score = test_loss
                        #print("current_best_score:", self.currnet_best_score)
                    perturbed_loss = test_loss + np.random.normal(0, self.nu* self.currnet_best_score *(1 - hp_budget))
                else:
                    if test_loss > self.currnet_best_score:
                        self.currnet_best_score = test_loss

                    perturbed_loss = (test_loss + np.random.normal(0, self.nu* (100.0- test_loss) *(1 - hp_budget)))/100.0
                 

                self.net.add_data((config_encoding, hp_budget), perturbed_loss )
                
                # train network
                if self.net.data_size % self.train_step_size ==0:
                    predict_score = self.net.predict((config_encoding, hp_budget)).item()
                    self.net.train()

                    
                  # ---------------- ban ------------------
                
                
                
                task_info = self.terminator.on_task_report(trial_id, result)
                task_continues = task_info["task_continues"]
                milestone_reached = task_info["milestone_reached"]
                if cost_and_promotion:
                    if milestone_reached:
                        # Trial reached milestone and will pause there: Update
                        # cost offset
                        if self._cost_attr is not None:
                            self._cost_offset[trial_id] = result[
                                self._total_cost_attr()
                            ]
                    elif task_info.get("ignore_data", False):
                        if self._cost_offset[trial_id] > 0:
                            logger.info(
                                f"trial_id {trial_id}: Resumed trial seems to have been "
                                + "started from scratch (no checkpointing?), so we erase "
                                + "the cost offset."
                            )
                        self._cost_offset[trial_id] = 0

                do_update = self._update_searcher(trial_id, config, result, task_info)
                resource = int(result[self._resource_attr])
                self._active_trials[trial_id].update(
                    {
                        "time_stamp": time_since_start,
                        "reported_result": {
                            self.metric: result[self.metric],
                            self._resource_attr: resource,
                        },
                        "keep_case": milestone_reached,
                    }
                )
                if do_update:
                    largest_update_resource = self._active_trials[trial_id][
                        "largest_update_resource"
                    ]
                    if largest_update_resource is None:
                        largest_update_resource = resource - 1
                    assert largest_update_resource <= resource, (
                        f"Internal error (trial_id {trial_id}): "
                        + f"on_trial_result called with resource = {resource}, "
                        + f"but largest_update_resource = {largest_update_resource}"
                    )
                    if resource == largest_update_resource:
                        do_update = False  # Do not update again
                    else:
                        self._active_trials[trial_id][
                            "largest_update_resource"
                        ] = resource
                if not task_continues:
                    if (not self.does_pause_resume()) or resource >= self.max_t:
                        trial_decision = SchedulerDecision.STOP
                        act_str = "Terminating"
                    else:
                        trial_decision = SchedulerDecision.PAUSE
                        act_str = "Pausing"
                    self._cleanup_trial(trial_id)
                if debug_log is not None:
                    if not task_continues:
                        logger.info(
                            f"trial_id {trial_id}: {act_str} evaluation "
                            f"at {resource}"
                        )
                    elif milestone_reached:
                        msg = f"trial_id {trial_id}: Reaches {resource}, continues"
                        next_milestone = task_info.get("next_milestone")
                        if next_milestone is not None:
                            msg += f" to {next_milestone}"
                        logger.info(msg)
            self.searcher.on_trial_result(
                trial_id, config, result=result, update=do_update
            )
        # Extra info in debug mode
        log_msg = f"trial_id {trial_id} (metric = {result[self.metric]:.3f}"
        for k, is_float in ((self._resource_attr, False), ("elapsed_time", True)):
            if k in result:
                if is_float:
                    log_msg += f", {k} = {result[k]:.2f}"
                else:
                    log_msg += f", {k} = {result[k]}"
        log_msg += f"): decision = {trial_decision}"
        logger.debug(log_msg)
        #pdb.set_trace()
        return trial_decision

    def on_trial_remove(self, trial: Trial):
        #print("remove trail", trial)
        self._cleanup_trial(str(trial.trial_id))

    def on_trial_complete(self, trial: Trial, result: Dict):
        # Check whether searcher was already updated based on `result`
        trial_id = str(trial.trial_id)
        largest_update_resource = self._active_trials[trial_id][
            "largest_update_resource"
        ]
        if largest_update_resource is not None:
            resource = int(result[self._resource_attr])
            if resource > largest_update_resource:
                super().on_trial_complete(trial, result)
        # Remove pending evaluations, in case there are still some
        self.searcher.cleanup_pending(trial_id)
        self._cleanup_trial(trial_id)

    def _get_paused_trials(self) -> Dict:
        rem_keys = {"config", "bracket", "running"}
        return {
            k: {k2: v[k2] for k2 in rem_keys}
            for k, v in self._active_trials.items()
            if not v["running"]
        }

    def does_pause_resume(self) -> bool:
        """
        :return: Is this variant doing pause and resume scheduling, in the
            sense that trials can be paused and resumed later?
        """
        return self.scheduler_type != "stopping"

    def _extend_search_options(self, search_options: Dict) -> Dict:
        # Note: Needs self.scheduler_type to be set
        scheduler = "hyperband_{}".format(self.scheduler_type)
        result = dict(
            search_options, scheduler=scheduler, resource_attr=self._resource_attr
        )
        # Cost attribute: For promotion-based, cost needs to be accumulated
        # for each trial
        cost_attr = self._total_cost_attr()
        if cost_attr is not None:
            result["cost_attr"] = cost_attr
        return result

    def _total_cost_attr(self) -> Optional[str]:
        if self._cost_attr is None:
            return None
        elif self.does_pause_resume():
            return "total_" + self._cost_attr
        else:
            return self._cost_attr
        
        
        
