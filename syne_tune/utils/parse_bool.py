def parse_bool(val: str) -> bool:
    val = val.upper()
    if val == "TRUE":
        return True
    else:
        assert val == "FALSE", f"val = '{val}' is not a boolean value"
        return False
