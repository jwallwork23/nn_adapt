import argparse


def _check_positive(value, typ):
    tvalue = typ(value)
    if tvalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive {typ} value")
    return tvalue


positive_float = lambda value: _check_positive(value, float)
positive_int = lambda value: _check_positive(value, int)


def _check_nonnegative(value, typ):
    tvalue = typ(value)
    if tvalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive {typ} value")
    return tvalue


nonnegative_float = lambda value: _check_nonnegative(value, float)
nonnegative_int = lambda value: _check_nonnegative(value, int)


def _check_in_range(value, typ, l, u):
    tvalue = typ(value)
    if not (tvalue >= l and tvalue <= u):
        raise argparse.ArgumentTypeError(f"{value} is not bounded by {(l, u)}")
    return tvalue


def bounded_float(l, u):
    def chk(value):
        return _check_in_range(value, float, l, u)

    return chk
