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
from typing import Optional, Dict, Any, List
import logging
import numpy as np

from syne_tune.optimizer.schedulers.searchers import BaseSearcher
from syne_tune.optimizer.schedulers.searchers.utils.exclusion_list import ExclusionList
from syne_tune.optimizer.schedulers.searchers.utils import (
    HyperparameterRanges,
    make_hyperparameter_ranges,
)

logger = logging.getLogger(__name__)


MAX_RETRIES = 100


def extract_random_seed(**kwargs) -> (int, Dict[str, Any]):
    key = "random_seed_generator"
    generator = kwargs.get(key)
    if generator is not None:
        random_seed = generator()
    else:
        key = "random_seed"
        random_seed = kwargs.get(key)
        if random_seed is None:
            random_seed = 31415927
            key = None
    _kwargs = {k: v for k, v in kwargs.items() if k != key}
    return random_seed, _kwargs


def sample_random_configuration(
    hp_ranges: HyperparameterRanges,
    random_state: np.random.RandomState,
    exclusion_list: Optional[ExclusionList] = None,
) -> Optional[Dict[str, Any]]:
    """
    Samples a configuration from ``config_space`` at random.

    :param hp_ranges: Used for sampling configurations
    :param random_state: PRN generator
    :param exclusion_list: Configurations not to be returned
    :return: New configuration, or ``None`` if configuration space has been
        exhausted
    """
    new_config = None
    no_exclusion = exclusion_list is None
    if no_exclusion or not exclusion_list.config_space_exhausted():
        for _ in range(MAX_RETRIES):
            _config = hp_ranges.random_config(random_state)
            if no_exclusion or not exclusion_list.contains(_config):
                new_config = _config
                break
    return new_config


class StochasticSearcher(BaseSearcher):
    """
    Base class of searchers which use random decisions. Creates the
    ``random_state`` member, which must be used for all random draws.

    Making proper use of this interface allows us to run experiments with
    control of random seeds, e.g. for paired comparisons or integration testing.

    Additional arguments on top of parent class :class:`BaseSearcher`:

    :param random_seed_generator: If given, random seed is drawn from there
    :type random_seed_generator: :class:`~syne_tune.optimizer.schedulers.random_seeds.RandomSeedGenerator`, optional
    :param random_seed: Used if ``random_seed_generator`` is not given.
    :type random_seed: int, optional
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        points_to_evaluate: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        super().__init__(
            config_space,
            metric=metric,
            points_to_evaluate=points_to_evaluate,
            mode=kwargs.get("mode", "min"),
        )
        random_seed, _ = extract_random_seed(**kwargs)
        self.random_state = np.random.RandomState(random_seed)

    def get_state(self) -> Dict[str, Any]:
        return dict(
            super().get_state(),
            random_state=self.random_state.get_state(),
        )

    def _restore_from_state(self, state: Dict[str, Any]):
        super()._restore_from_state(state)
        self.random_state.set_state(state["random_state"])

    def set_random_state(self, random_state: np.random.RandomState):
        self.random_state = random_state

    def _filter_points_to_evaluate(
        self,
        restrict_configurations: List[Dict[str, Any]],
        hp_ranges: HyperparameterRanges,
        allow_duplicates: bool,
    ) -> List[Dict[str, Any]]:
        """
        Used to support ``restrict_configurations`` in subclasses. Configs in
        ``_points_to_evaluate`` are removed if not in ``restrict_configurations``.
        If ``allow_duplicates == False``, entries in ``_points_to_evaluate`` are
        removed from ``restrict_configurations``. The filtered list
        ``restrict_configurations`` is returned.

        :param restrict_configurations: See above
        :param hp_ranges: Used to map configs to match strings
        :param allow_duplicates: See above
        :return: Filtered ``restrict_configurations``
        """
        assert len(restrict_configurations) > 0
        remove_p2e = []
        remove_rc = []
        matchstr_to_pos = {
            hp_ranges.config_to_match_string(config): pos
            for pos, config in enumerate(restrict_configurations)
        }
        for pos_p2e, config in enumerate(self._points_to_evaluate):
            pos_rc = matchstr_to_pos.get(hp_ranges.config_to_match_string(config))
            if pos_rc is None:
                # Entry in ``points_to_evaluate`` not in
                # ``restrict_configurations``, has to be removed
                remove_p2e.append(pos_p2e)
            elif not allow_duplicates:
                # Entry in ``points_to_evaluate`` can be removed from
                # ``restrict_configurations``, because will be suggested at
                # the beginning
                remove_rc.append(pos_rc)
        if remove_p2e:
            msg_parts = [
                "These configs are in points_to_evaluate, but not in "
                "restrict_configurations. They are removed:"
            ]
            remove_p2e = set(remove_p2e)
            new_p2e = []
            for pos, config in enumerate(self._points_to_evaluate):
                if pos in remove_p2e:
                    msg_parts.append(str(config))
                else:
                    new_p2e.append(config)
            self._points_to_evaluate = new_p2e
            logger.warning("\n".join(msg_parts))
        if remove_rc:
            remove_rc = set(remove_rc)
            restrict_configurations = [
                config
                for pos, config in enumerate(restrict_configurations)
                if pos not in remove_rc
            ]
        return restrict_configurations


class StochasticAndFilterDuplicatesSearcher(StochasticSearcher):
    """
    Base class for searchers with the following properties:

    * Random decisions use common :attr:`random_state`
    * Maintains exclusion list to filter out duplicates in
      :meth:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher.get_config`
      if ``allows_duplicates == False`. If this is ``True``, duplicates are not
      filtered, and the exclusion list is used only to avoid configurations of
      failed trials.
    * If ``restrict_configurations`` is given, this is a list of configurations,
      and the searcher only suggests configurations from there. If
      ``allow_duplicates == False``, entries are popped off this list once
      suggested.
      ``points_to_evaluate`` is filtered to only contain entries in this set.

    In order to make use of these features:

    * Reject configurations in :meth:`get_config` if :meth:`should_not_suggest`
      returns ``True``.
      If the configuration is drawn at random, use :meth:`_get_random_config`,
      which incorporates this filtering
    * Implement :meth:`_get_config` instead of :meth:`get_config`. The latter
      adds the new config to the exclusion list if ``allow_duplicates == False``

    Note: Not all searchers which filter duplicates make use of this class.

    Additional arguments on top of parent class :class:`StochasticSearcher`:

    :param allow_duplicates: See above. Defaults to ``False``
    :param restrict_configurations: See above, optional
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        points_to_evaluate: Optional[List[Dict[str, Any]]] = None,
        allow_duplicates: Optional[bool] = None,
        restrict_configurations: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        super().__init__(
            config_space, metric=metric, points_to_evaluate=points_to_evaluate, **kwargs
        )
        self._hp_ranges = make_hyperparameter_ranges(config_space)
        if allow_duplicates is None:
            allow_duplicates = False
        self._allow_duplicates = allow_duplicates
        # Used to avoid returning the same config more than once. If
        # ``allow_duplicates == True``, this is used to block failed trials
        self._excl_list = ExclusionList(self._hp_ranges)
        # Maps ``trial_id`` to configuration. This is used to blacklist
        # configurations whose trial has failed (only if
        # `allow_duplicates == True``)
        self._config_for_trial_id = dict() if allow_duplicates else None
        # Assign ``_restrict_configurations`` and filter ``_points_to_evaluate``
        # accordingly
        if restrict_configurations is None:
            self._restrict_configurations = None
            self._rc_returned_pos = None
        else:
            self._restrict_configurations = self._filter_points_to_evaluate(
                restrict_configurations, self._hp_ranges, self._allow_duplicates
            )
            self._rc_returned_pos = set()

    @property
    def allow_duplicates(self) -> bool:
        return self._allow_duplicates

    def should_not_suggest(self, config: Dict[str, Any]) -> bool:
        """
        :param config: Configuration
        :return: :meth:`get_config` should not suggest this configuration?
        """
        return self._excl_list.contains(config)

    def _get_config(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Child classes implement this instead of :meth:`get_config`.
        """
        raise NotImplementedError

    def get_config(self, **kwargs) -> Optional[Dict[str, Any]]:
        new_config = self._get_config(**kwargs)
        if not self._allow_duplicates and new_config is not None:
            self._excl_list.add(new_config)
            if self._restrict_configurations is not None and self._rc_returned_pos:
                # If ``new_config`` has been returned by :meth:`_get_random_config`,
                # remove it from the list.
                # This is a compromise. We could search ``new_config`` in all of
                # ``_restrict_configurations``, but this is too expensive
                ms_new = self._hp_ranges.config_to_match_string(new_config)
                for pos in self._rc_returned_pos:
                    ms_rc = self._hp_ranges.config_to_match_string(
                        self._restrict_configurations[pos]
                    )
                    if ms_rc == ms_new:
                        self._restrict_configurations.pop(pos)
                        break
                self._rc_returned_pos = set()  # Reset
        return new_config

    def _get_random_config(
        self, exclusion_list: Optional[ExclusionList] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Child classes should use this helper method in order to draw a configuration at
        random.

        :param exclusion_list: Configurations to be avoided. Defaults to ``self._excl_list``
        :return: Configuration drawn at random, or ``None`` if the configuration space
            has been exhausted w.r.t. ``exclusion_list``
        """
        if exclusion_list is None:
            exclusion_list = self._excl_list
        if self._restrict_configurations is not None:
            return self._get_random_config_from_restrict_configurations(exclusion_list)
        else:
            return sample_random_configuration(
                hp_ranges=self._hp_ranges,
                random_state=self.random_state,
                exclusion_list=exclusion_list,
            )

    def _get_random_config_from_restrict_configurations(
        self, exclusion_list: ExclusionList
    ) -> Optional[Dict[str, Any]]:
        config = None
        if self._restrict_configurations:
            for _ in range(MAX_RETRIES):
                pos = self.random_state.randint(
                    low=0, high=len(self._restrict_configurations)
                )
                config = self._restrict_configurations[pos]
                if exclusion_list.contains(config):
                    config = None
                    continue  # Try again
                if not self.allow_duplicates:
                    # Mark for (potential) later removal in :meth:`get_config`.
                    # We cannot remove the config here, because
                    # :meth:`_get_random_config` can be called for other reasons
                    self._rc_returned_pos.add(pos)
                break  # Leave loop
        return config

    def register_pending(
        self,
        trial_id: str,
        config: Optional[Dict[str, Any]] = None,
        milestone: Optional[int] = None,
    ):
        super().register_pending(trial_id, config, milestone)
        if self._allow_duplicates and trial_id not in self._config_for_trial_id:
            if config is not None:
                self._config_for_trial_id[trial_id] = config
            else:
                logger.warning(
                    f"register_pending called for trial_id {trial_id} without passing config"
                )

    def evaluation_failed(self, trial_id: str):
        super().evaluation_failed(trial_id)
        if self._allow_duplicates and trial_id in self._config_for_trial_id:
            # Blacklist this configuration
            self._excl_list.add(self._config_for_trial_id[trial_id])

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["excl_list"] = self._excl_list.get_state()
        if self._allow_duplicates:
            state["config_for_trial_id"] = self._config_for_trial_id
        if self._restrict_configurations is not None:
            state["restrict_configurations"] = self._restrict_configurations
        return state

    def _restore_from_state(self, state: Dict[str, Any]):
        super()._restore_from_state(state)
        self._excl_list = ExclusionList(self._hp_ranges)
        self._excl_list.clone_from_state(state["excl_list"])
        if self._allow_duplicates:
            self._config_for_trial_id = state["config_for_trial_id"]
        k = "restrict_configurations"
        if k in state:
            self._restrict_configurations = state[k]
        else:
            self._restrict_configurations = None
