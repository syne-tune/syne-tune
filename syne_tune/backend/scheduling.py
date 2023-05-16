import functools
import logging
import time


def backoff(errorname: str, ntimes_resource_wait: int = 100, length2sleep: float = 600):
    """
    Decorator that back offs for a fixed about of s after a given error is detected
    """

    def errorcatch(some_function):
        @functools.wraps(some_function)
        def wrapper(*args, **kwargs):
            for idx in range(ntimes_resource_wait):
                try:
                    return some_function(*args, **kwargs)
                except Exception as e:
                    if not e.__class__.__name__ == errorname:
                        raise (e)

                logging.info(
                    f"{errorname} detected when calling <{some_function.__name__}>, waiting {length2sleep / 60} minutes before retring"
                )
                time.sleep(length2sleep)
                continue

        return wrapper

    return errorcatch
