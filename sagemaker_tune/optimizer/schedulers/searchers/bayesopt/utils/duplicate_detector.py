from abc import ABC, abstractmethod

from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import Configuration
from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.common \
    import ExclusionList


class DuplicateDetector(ABC):
    @abstractmethod
    def contains(self, existing_candidates: ExclusionList,
                 new_candidate: Configuration) -> bool:
        pass


class DuplicateDetectorNoDetection(DuplicateDetector):
    def contains(self, existing_candidates: ExclusionList,
                 new_candidate: Configuration) -> bool:
        return False  # no duplicate detection at all


class DuplicateDetectorIdentical(DuplicateDetector):
    def contains(self, existing_candidates: ExclusionList,
                 new_candidate: Configuration) -> bool:
        return existing_candidates.contains(new_candidate)
