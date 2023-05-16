import datetime

import pytest

from syne_tune.backend.scheduling import backoff


@backoff(
    errorname="AttributeError",
    ntimes_resource_wait=10,
    length2sleep=0.01
)
def errorfunction(starttime: datetime.datetime):
    if starttime is None:
        raise ValueError

    # This will fail if the function tries to finish less than 0.1s after being called
    time_since_start =  datetime.datetime.now() - starttime
    if time_since_start < datetime.timedelta(microseconds=int(1e5)):
        raise AttributeError

    return True


def test_backoff_completes():
    errorfunction(datetime.datetime.now())

def test_backoff_failure():
    with pytest.raises(ValueError) as e_info:
        errorfunction(None)