from typing import Any

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series


def to_numpy(x: Any) -> ndarray:
    if isinstance(x, DataFrame) or isinstance(x, Series):
        return x.to_numpy()
    elif isinstance(x, ndarray):
        return x
    elif isinstance(x, list):
        try:
            return np.array(x)
        except Exception as e:
            raise ValueError("Cannot create NumPy array from `x`.") from e
    else:
        raise ValueError("Unsupported type for predictor `x`")


def array_indexer(array: ndarray, sample_dim: int, idx: ndarray) -> ndarray:
    """Used to index into a specific position programmatically.

    :meta private:
    """
    colons = [":" for _ in range(array.ndim)]
    colons[sample_dim] = "idx"
    idx_string = f"{','.join(colons)}"
    return eval(f"array[{idx_string}]")
