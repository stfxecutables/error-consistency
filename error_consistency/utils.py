import numpy as np

from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.cluster import KMeans

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from typing import cast, no_type_check
from typing_extensions import Literal


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
