

def dict_get(params: dict, key: str, default):
    """
    Returns `params[key]` if this exists and is not None, and `default` otherwise.
    Note that this is not the same as `params.get(key, default)`. Namely, if `params[key]`
    is equal to None, this would return None, but this method returns `default`.

    This function is particularly helpful when dealing with a dict returned by
    :class:`argparse.ArgumentParser`. Whenever `key` is added as argument to the parser,
    but a value is not provided, this leads to `params[key] = None`.

    """
    v = params.get(key)
    return default if v is None else v
