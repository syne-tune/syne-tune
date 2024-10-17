from typing import List


class MultiFidelitySchedulerMixin:
    """
    Declares properties which are required for multi-fidelity schedulers.
    """

    @property
    def resource_attr(self) -> str:
        """
        :return: Name of resource attribute in reported results
        """
        raise NotImplementedError

    @property
    def max_resource_level(self) -> int:
        """
        :return: Maximum resource level
        """
        raise NotImplementedError

    @property
    def rung_levels(self) -> List[int]:
        """
        :return: Rung levels (positive int; increasing), may or may not
            include ``max_resource_level``
        """
        raise NotImplementedError

    @property
    def searcher_data(self) -> str:
        """
        :return: Relevant only if a model-based searcher is used.
            Example: For NN tuning and ``resource_attr == "epoch"``, we receive
            a result for each epoch, but not all epoch values are also rung
            levels. ``searcher_data`` determines which of these results are
            passed to the searcher. As a rule, the more data the searcher
            receives, the better its fit, but also the more expensive
            :meth:`get_config` may become. Choices:

            * "rungs": Only results at rung levels. Cheapest
            * "all": All results. Most expensive
            * "rungs_and_last": Results at rung levels plus last recent one.
              Not available for all multi-fidelity schedulers
        """
        raise NotImplementedError

    @property
    def num_brackets(self) -> int:
        """
        :return: Number of brackets (i.e., rung level systems). If the scheduler
            does not use brackets, it has to return 1
        """
        raise NotImplementedError
