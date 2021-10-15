import logging

from sagemaker.estimator import Framework
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT

logger = logging.getLogger(__name__)


class CustomFramework(Framework):

    __framework_name__ = "customframework"

    LATEST_VERSION = '0.1'

    def __init__(
        self,
        entry_point,
        image_uri: str,
        source_dir=None,
        hyperparameters=None,
        **kwargs
    ) -> None:
        super(CustomFramework, self).__init__(
            str(entry_point),
            source_dir,
            hyperparameters,
            image_uri=image_uri,
            **kwargs
        )

    def create_model(
        self,
        model_server_workers=None,
        role=None,
        vpc_config_override=VPC_CONFIG_DEFAULT
    ):
        # required to allow this object instantiation
        raise NotImplementedError()
