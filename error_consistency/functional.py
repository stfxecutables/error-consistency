from __future__ import annotations

from functools import reduce
from itertools import combinations
from multiprocessing import cpu_count
from typing import List, Tuple, Union
from warnings import warn

import numpy as np
from numba import prange, jit
from numpy import ndarray
from pandas import DataFrame, Series
from typing import Optional
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal
from tqdm import tqdm

from error_consistency.utils import to_numpy

UNION_OPTIONS = [0, 1, "nan", "drop", "warn", "error"]
ArrayLike = Union[ndarray, DataFrame, Series]
UnionHandling = Literal[0, 1, "nan", "drop", "warn", "error"]
"""UnionHandling"""
_UnionHandling = Literal["0", "1", "nan", "drop", "warn", "error"]

Normalization = Literal[None, "length", "intersection"]
_Normalization = Literal["", "length", "intersection"]


@jit(nopython=True, parallel=True, cache=True)
def error_consistencies_numba(
    y_errs: ndarray, empty_unions: _UnionHandling = "nan", norm: _Normalization = ""
) -> ndarray:
    L = len(y_errs)
    matrix = np.nan * np.ones((L, L))
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
                    matrix[i, j] = matrix[j, i] = np.nan
                    continue
                elif empty_unions == "0":
                    matrix[i, j] = matrix[j, i] = 0.0
                    continue
                elif empty_unions == "1":
                    matrix[i, j] = matrix[j, i] = 1.0
                    continue
                else:
                    local_union = -0.00000001
            if norm == "length":
                local_union = len(err_i)
            score = np.sum(err_i & err_j) / local_union
            matrix[i, j] = matrix[j, i] = score
    return matrix


def error_consistencies_slow(
    y_errs: List[ndarray], empty_unions: _UnionHandling = "0", norm: _Normalization = ""
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
                elif empty_unions == "0":
                    consistencies.append(0)
                    matrix[i, j] = matrix[j, i] = 0
                    continue
                elif empty_unions == "1":
                    consistencies.append(1)
                    matrix[i, j] = matrix[j, i] = 1
                    continue
                elif empty_unions == "warn":
                    warn(f"Empty union in comparison ({i}, {j})")
                    consistencies.append(np.nan)
                    continue
                elif empty_unions == "error":
                    raise ZeroDivisionError(f"Empty union in comparison ({i}, {j})")
                else:
                    raise ValueError("Unreachable!")
            if norm == "length":
                local_union = len(err_i)
            score = np.sum(err_i & err_j) / local_union
            consistencies.append(score)
            matrix[i, j] = matrix[j, i] = score
        if i % 100 == 0:
            pbar.set_description(desc.format(np.mean(consistencies)))
        pbar.update()
    pbar.close()
    return np.array(consistencies), matrix


def loo_loop(args: Tuple[ndarray, _UnionHandling, _Normalization]) -> Optional[ndarray]:
    lsets, empty_unions, norm = args

    numerator = np.sum(intersection(list(lsets)))
    denom = np.sum(union(list(lsets)))
    if denom == 0:
        if empty_unions == "drop":
            return None
        elif empty_unions == "nan":
            return None
        elif empty_unions == "0":
            return 0.0
        elif empty_unions == "1":
            return 1.0
        elif empty_unions == "warn":
            warn("Empty union in leave-one-out error sets.")
            return None
        elif empty_unions == "error":
            raise ZeroDivisionError("Empty union in leave-one-out error sets.")
        else:
            raise ValueError("Unreachable!")
    if norm == "length":
        denom = len(lsets[0])
    return numerator / denom


def loo_consistencies(
    y_errs: List[ndarray],
    empty_unions: _UnionHandling = "0",
    parallel: bool = False,
    norm: _Normalization = "",
) -> ndarray:
    L = len(y_errs)
    loo_sets = combinations(y_errs, L - 1)  # leave-one out
    if parallel:
        args = [(lsets, empty_unions, norm) for lsets in loo_sets]
        # https://stackoverflow.com/questions/53751050/python-multiprocessing-understanding-logic-behind-chunksize
        chunksize = divmod(L, cpu_count() * 20)[0]
        chunksize = 1 if chunksize < 1 else chunksize
        consistencies = process_map(
            loo_loop,
            args,
            max_workers=cpu_count(),
            chunksize=chunksize,
            desc="Computing leave-one-out error consistencies",
            total=L,
        )
        if empty_unions == "drop":
            consistencies = np.array(filter(lambda c: c is None, consistencies))
        if empty_unions == "nan":
            consistencies = np.array(
                map(lambda c: np.nan if c is None else c, consistencies)  # type: ignore
            )
    else:
        consistencies = []
        for lset in tqdm(loo_sets, total=L, desc="Computing leave-one-out error consistencies"):
            numerator = np.sum(intersection(list(lset)))
            denom = np.sum(union(list(lset)))
            if denom == 0:
                if empty_unions == "drop":
                    continue
                elif empty_unions == "nan":
                    consistencies.append(np.nan)
                    continue
                elif empty_unions == "0":
                    consistencies.append(0.0)
                    continue
                elif empty_unions == "1":
                    consistencies.append(1.0)
                    continue
                elif empty_unions == "warn":
                    warn("Empty union in leave-one-out error sets.")
                    return None
                elif empty_unions == "error":
                    raise ZeroDivisionError("Empty union in leave-one-out error sets.")
                else:
                    raise ValueError("Unreachable!")
            if norm == "length":
                denom = len(lset[0])
            consistencies.append(numerator / denom)
    return np.array(consistencies)


def get_y_error(y_pred: ndarray, y_true: ndarray, sample_dim: int = 0) -> ndarray:
    """Returns the

    Returns
    -------
    error_set: ndarray
        The boolean array with length `n_samples` where y_pred != y_true. For one-hot or dummy `y`,
        this is still computed such that the length of the returned array is `n_samples`.
    """
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
    empty_unions: UnionHandling = 0,
    loo_parallel: bool = False,
    turbo: bool = False,
    log_progress: bool = False,
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """Get the error consistency for a list of predictions.

    Parameters
    ----------
    y_preds: List[ndarray]
        A list of numpy arrays of predictions, all on the same test set.

    y_true: ndarray
        The true values against which each of `y_preds` will be compared.

    sample_dim: int = 0
        The dimension along which samples are indexed for `y_true` and each array of `y_preds`.

    empty_unions: UnionHandling = 0
        When computing the pairwise consistency or leave-one-out consistency on small or
        simple datasets, it can be the case that the union of the error sets is empty (e.g. if no
        prediction errors are made). In this case the intersection over union is 0/0, which is
        undefined.

        * If `0` (default), the consistency for that collection of error sets is set to zero.
        * If `1`, the consistency for that collection of error sets is set to one.
        * If "nan", the consistency for that collection of error sets is set to `np.nan`.
        * If "drop", the `consistencies` array will not include results for that collection,
          but the consistency matrix will include `np.nans`.
        * If "error", an empty union will cause a `ZeroDivisionError`.
        * If "warn", an empty union will print a warning (probably a lot).

    loo_parallel: bool = False
        If True, use multiprocessing to parallelize the computation of the leave-one-out error
        consistencies.

    turbo: bool = False
        If True, use Numba-accelerated error consistency calculation.

    log_progress: bool = False
        If True, show a progress bar when computing the leave-one-out error consistencies.

    Returns
    -------
    consistencies: ndarray
        An array of the computed consistency values. Length will depend on `empty_unions`.

    matrix: ndarray
        An array of size ``(N, N)`` of the pairwise consistency values (IOUs) where
        `N = len(y_preds)`, and where entry ``(i, j)`` is the pairwise IOU for predictions ``i`` and
        predictions ``j``.

    intersection: ndarray
        The total intersection of all error sets. When nonzero, can be useful for identifying
        unpredictable samples.

    union: ndarray
        The total union of all error sets. Will almost always be non-empty except for trivial
        datasets, and thus computing ``np.sum(intersection) / np.sum(union)`` gives something like a
        lower bound on the consistencies.

    loo_consistencies: ndarray
        The IOUs or consistencies computed from applying the union and intesection operations over
        all combinations of `y_preds` of size `len(y_preds) - 1`. Sort of a symmetric counterpart
        to the default pairwise consistency.
    """
    # y_preds must be (reps, n_samples), y_true must be (n_samples,) or (n_samples, 1) or
    # (n_samples, n_features)
    if empty_unions not in UNION_OPTIONS:
        raise ValueError(
            f"Invalid option for handling empty unions. Must be one of {UNION_OPTIONS}"
        )
    if empty_unions in [0, 1]:
        empty_unions = str(empty_unions)  # type: ignore

    if not isinstance(y_preds, list) or len(y_preds) < 2:  # type: ignore
        raise ValueError("`y_preds` must be a list of predictions with length > 1.")

    y_preds = list(map(to_numpy, y_preds))
    y_errs = [get_y_error(y_pred, y_true, sample_dim) for y_pred in y_preds]

    unpredictable_set = intersection(y_errs)
    predictable_set = union(y_errs)
    if log_progress:
        print("Computing Leave-One-Out error consistencies ... ", end="", flush=True)
    loocs = loo_consistencies(y_errs, empty_unions, loo_parallel)  # type: ignore
    if log_progress:
        print("done.")

    if turbo:
        if log_progress:
            print("Computing pairwise error consistencies ... ", end="", flush=True)
        # we don't do the list in the numba call because this won't parallelize as well
        # i.e. you need the index into the consistencies array because you can't append
        matrix = error_consistencies_numba(np.array(y_errs), empty_unions)
        cs = matrix[np.triu_indices_from(matrix, 1)].ravel()
        consistencies = cs[~np.isnan(cs)] if empty_unions == "drop" else cs
        if log_progress:
            print("done.")
    else:
        consistencies, matrix = error_consistencies_slow(y_errs, empty_unions)
    return (np.array(consistencies), matrix, unpredictable_set, predictable_set, loocs)
