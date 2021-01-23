from typing import Any, Dict, List, Optional, Tuple, Union
from typing import cast, no_type_check
from typing_extensions import Literal

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series

from warnings import warn

from error_consistency.utils import to_numpy

ArrayLike = Union[ndarray, DataFrame, Series]


def error_consistencies(
    y_preds: List[ndarray], y_true: ndarray, sample_dim: int = 0
) -> Tuple[List[ndarray], ndarray]:
    """Get the error consistency for a list of predictions."""
    # y_preds must be (reps, n_samples), y_true must be (n_samples,) or (n_samples, 1) or
    # (n_samples, n_features)
    if not isinstance(y_preds, list) or len(y_preds) < 2:
        raise ValueError("`y_preds` must be a list of predictions with length > 1.")

    y_preds = list(map(to_numpy, y_preds))
    y_errs = []
    for y_pred in y_preds:
        y_errs.append(get_y_error(y_pred, y_true, sample_dim))

    consistencies, matrix = [], -np.ones([len(y_preds), len(y_preds)], dtype=float)
    for i, err_i in enumerate(y_errs):
        for j, err_j in enumerate(y_errs):
            if i >= j:
                continue
            score = np.sum(err_i & err_j) / np.sum(err_i | err_j)
            consistencies.append(score)
            matrix[i, j] = score

    return consistencies, matrix


def get_y_error(y_pred: ndarray, y_true: ndarray, sample_dim: int = 0) -> ndarray:
    if y_pred.ndim != y_true.ndim:
        raise ValueError("`y_pred` and `y_true` must have the same dimensionality.")
    sample_dim = int(np.abs(sample_dim))
    if sample_dim not in [0, 1]:
        raise ValueError("`sample_dim` must be an integer in the set {0, 1, -1}")

    if y_pred.ndim == 1:
        return y_pred.ravel().astype(int) == y_true.ravel().astype(int)
    if y_pred.ndim == 2:
        label_dim = 1 - sample_dim
        return ~np.alltrue(y_pred.astype(int) == y_true.astype(int), axis=label_dim)
    raise ValueError(
        "Error consistency only supported for label encoding, dummy coding, or one-hot encoding"
    )

