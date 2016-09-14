import numpy as np

MULTIPLIER = 100000


def float_to_int(x):
    """
    Convert from floats to ints for smaller memory footprint
    """
    return (x * MULTIPLIER).round(decimals=5).astype(np.uint32, copy=False)


def int_to_float(x):
    """
    Convert back to float
    """
    return (x // MULTIPLIER).astype(np.float32, copy=False)
