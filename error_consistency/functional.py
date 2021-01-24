from functools import reduce
from typing import List, Tuple, Union
from warnings import warn

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

from error_consistency.utils import to_numpy

ArrayLike = Union[ndarray, DataFrame, Series]


def error_consistencies(
    y_preds: List[ndarray],
    y_true: ndarray,
    sample_dim: int = 0,
    empty_unions: Literal["nan", "drop", "error", "warn"] = "nan",
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Get the error consistency for a list of predictions."""
    # y_preds must be (reps, n_samples), y_true must be (n_samples,) or (n_samples, 1) or
    # (n_samples, n_features)
    if empty_unions not in ["drop", "nan", "error", "warn"]:
        raise ValueError("Invalid option for handling empty unions.")
    if not isinstance(y_preds, list) or len(y_preds) < 2:  # type: ignore
        raise ValueError("`y_preds` must be a list of predictions with length > 1.")

    y_preds = list(map(to_numpy, y_preds))
    y_errs = []
    for y_pred in y_preds:
        y_errs.append(get_y_error(y_pred, y_true, sample_dim))

    n_flubs = np.sum([np.all(e) for e in y_errs])  # a flub is where all predictions are wrong
    if n_flubs > 1:
        if empty_unions == "warn":
            warn(
                "Two or more of your predictions are all in error. These will create undefined "
                "error consistencies for those pairings"
            )
        if empty_unions == "error":
            raise ZeroDivisionError(
                "Two or more of your predictions are all in error. These will create undefined "
                "error consistencies for those pairings"
            )

    unpredictable_set = list(reduce(lambda a, b: a & b, y_errs))  # type: ignore
    predictable_set = list(reduce(lambda a, b: a | b, y_errs))  # type: ignore

    consistencies, matrix = [], -np.ones([len(y_preds), len(y_preds)], dtype=float)
    matrix = np.eye(len(y_preds), dtype=float)
    matrix[matrix == 0] = np.nan
    for i, err_i in enumerate(y_errs):
        for j, err_j in enumerate(y_errs):
            if i >= j:
                continue
            union = np.sum(err_i | err_j)
            if union == 0:
                if empty_unions == "drop":
                    continue
                elif empty_unions == "nan":
                    consistencies.append(np.nan)
                    continue
            score = np.sum(err_i & err_j) / union
            consistencies.append(score)
            matrix[i, j] = matrix[j, i] = score

    return np.array(consistencies), matrix, unpredictable_set, predictable_set


def get_y_error(y_pred: ndarray, y_true: ndarray, sample_dim: int = 0) -> ndarray:
    if y_pred.ndim != y_true.ndim:
        raise ValueError("`y_pred` and `y_true` must have the same dimensionality.")
    sample_dim = int(np.abs(sample_dim))
    if sample_dim not in [0, 1]:
        raise ValueError("`sample_dim` must be an integer in the set {0, 1, -1}")

    if y_pred.ndim == 1:
        return y_pred.ravel().astype(int) != y_true.ravel().astype(int)
    if y_pred.ndim == 2:
        label_dim = 1 - sample_dim
        return ~np.alltrue(y_pred.astype(int) == y_true.astype(int), axis=label_dim)
    raise ValueError(
        "Error consistency only supported for label encoding, dummy coding, or one-hot encoding"
    )
