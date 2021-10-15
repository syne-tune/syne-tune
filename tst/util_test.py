import tempfile

from sagemaker_tune.backend.local_backend import LocalBackend


def temporary_local_backend(entry_point: str):
    """
    :param entry_point:
    :return: a backend whose files are deleted after finishing to avoid side-effects. This is used in unit-tests.
    """
    with tempfile.TemporaryDirectory() as local_path:
        return LocalBackend(entry_point=entry_point).set_path(results_root=local_path)
