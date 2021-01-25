from functools import reduce
from itertools import combinations
from typing import List, Tuple, Union
from warnings import warn

import numpy as np
from numba import prange, jit
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal
from tqdm import tqdm

from error_consistency.utils import to_numpy

ArrayLike = Union[ndarray, DataFrame, Series]
UnionHandling = Literal["nan", "drop", "error", "warn", "zero", "+1"]


@jit(nopython=True, parallel=True, cache=True)
def error_consistencies_numba(
    y_errs: ndarray, empty_unions: UnionHandling = "nan"
) -> Tuple[List, ndarray]:
    L = len(y_errs)
    consistencies = np.empty((L * (L - 1) // 2), dtype=np.float64)
    matrix = np.zeros((L, L))
    k = 0  # consistency index
    for i in prange(L):
        err_i = y_errs[i]
        for j in range(L):
            err_j = y_errs[j]
            if i == j:
                matrix[i, j] = 1.0
                continue
            if i > j:
                continue
            local_union = np.sum(err_i | err_j)
            if local_union == 0:
                if empty_unions == "drop":
                    continue
                elif empty_unions == "nan":
                    consistencies[k] = np.nan
                    k += 1
                    continue
                elif empty_unions == "zero":
                    consistencies[k] = 0.0
                    k += 1
                    matrix[i, j] = matrix[j, i] = 0.0
                    continue
                elif empty_unions == "+1":
                    local_union += 1.0
                else:
                    local_union = -0.00000001
            score = np.sum(err_i & err_j) / local_union
            consistencies[k] = score
            k += 1
            matrix[i, j] = matrix[j, i] = score
    return consistencies, matrix


def error_consistencies_slow(
    y_errs: List[ndarray], empty_unions: UnionHandling = "zero"
) -> Tuple[ndarray, ndarray]:
    consistencies = []
    matrix = np.eye(len(y_errs), dtype=float)
    matrix[matrix == 0] = np.nan
    desc = "Calculating consistencies. Running mean={:0.3f}"
    pbar = tqdm(total=len(y_errs), desc=desc.format(0))
    for i, err_i in enumerate(y_errs):
        for j, err_j in enumerate(y_errs):
            if i >= j:
                continue
            local_union = np.sum(union([err_i, err_j]))
            if local_union == 0:
                if empty_unions == "drop":
                    continue
                elif empty_unions == "nan":
                    consistencies.append(np.nan)
                    continue
                elif empty_unions == "zero":
                    consistencies.append(0)
                    matrix[i, j] = matrix[j, i] = 0
                    continue
                elif empty_unions == "+1":
                    local_union += 1
                else:
                    raise ValueError("Unreachable!")
            score = np.sum(err_i & err_j) / local_union
            consistencies.append(score)
            matrix[i, j] = matrix[j, i] = score
        if i % 100 == 0:
            pbar.set_description(desc.format(np.mean(consistencies)))
        pbar.update()
    pbar.close()
    return np.array(consistencies), matrix


def loo_consistencies(y_errs: List[ndarray], empty_unions: UnionHandling = "zero") -> ndarray:
    loo_sets = combinations(y_errs, len(y_errs) - 1)  # leave-one out
    consistencies = []
    for lset in loo_sets:
        numerator = np.sum(intersection(list(lset)))
        denom = np.sum(union(list(lset)))
        if denom == 0:
            if empty_unions == "drop":
                continue
            elif empty_unions == "nan":
                consistencies.append(np.nan)
                continue
            elif empty_unions == "zero":
                consistencies.append(0.0)
                continue
            elif empty_unions == "+1":
                denom += 1.0
            else:
                raise ValueError("Unreachable!")
        consistencies.append(numerator / denom)
    return np.array(consistencies)


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


def union(y_errs: List[ndarray]) -> ndarray:
    return reduce(lambda a, b: a | b, y_errs)


def intersection(y_errs: List[ndarray]) -> ndarray:
    return reduce(lambda a, b: a & b, y_errs)


def error_consistencies(
    y_preds: List[ndarray],
    y_true: ndarray,
    sample_dim: int = 0,
    empty_unions: UnionHandling = "zero",
    turbo: bool = False,
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """Get the error consistency for a list of predictions."""
    # y_preds must be (reps, n_samples), y_true must be (n_samples,) or (n_samples, 1) or
    # (n_samples, n_features)
    if empty_unions not in ["drop", "nan", "error", "warn", "zero", "+1"]:
        raise ValueError("Invalid option for handling empty unions.")
    if not isinstance(y_preds, list) or len(y_preds) < 2:  # type: ignore
        raise ValueError("`y_preds` must be a list of predictions with length > 1.")

    y_preds = list(map(to_numpy, y_preds))
    y_errs = [get_y_error(y_pred, y_true, sample_dim) for y_pred in y_preds]

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

    unpredictable_set = intersection(y_errs)
    predictable_set = union(y_errs)
    loocs = loo_consistencies(y_errs, empty_unions)

    if turbo:
        consistencies, matrix = error_consistencies_numba(np.array(y_errs), empty_unions)
    else:
        consistencies, matrix = error_consistencies_slow(y_errs, empty_unions)
    return (np.array(consistencies), matrix, unpredictable_set, predictable_set, loocs)
