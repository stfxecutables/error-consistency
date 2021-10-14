import timeit
from argparse import Namespace
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytest
from _pytest.capture import CaptureFixture
from numba import jit, prange
from numpy import ndarray
from numpy.random import Generator
from pandas import DataFrame
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from error_consistency.functional import error_consistencies, error_consistencies_numba, get_y_error

OUTDIR = Path(__file__).resolve().parent / "test_results"
NUM_SAMPLES = 10000


@jit(nopython=True, parallel=True)
def average_distance(y_errs: ndarray) -> ndarray:
    L = len(y_errs)
    matrix = np.nan * np.ones((L, L))
    for i in prange(L):
        err_i = y_errs[i]
        for j in range(L):
            err_j = y_errs[j]
            if i == j:
                # matrix[i, j] = 1.0
                continue
            if i > j:
                continue
            matrix[i, j] = np.mean(np.abs(err_i - err_j))
    return matrix


def get_sampling_probs(n: int, gamma: float) -> np.ndarray:
    """We just need an array of probabilities, e.g. must sum to 1, from some weights.
    Just compute a sloped line on domain [0, 1]
    """
    weights = np.linspace(0, 1, n) ** gamma
    return weights / weights.sum()


def n_correlated_errs(n: int, gamma: float, num_errors: int) -> np.ndarray:
    rng: Generator = np.random.default_rng()
    y_errs = []
    p = get_sampling_probs(n, gamma)
    for i in range(num_errors):
        k = rng.integers(0, n + 1)
        errs = np.concatenate([np.ones(k, dtype=bool), np.zeros(n - k, dtype=bool)])
        y_errs.append(rng.choice(errs, size=n, p=p, replace=True))
    y_errs = np.array(y_errs)
    matrix = error_consistencies_numba(y_errs, empty_unions="nan")
    cs = matrix[np.triu_indices_from(matrix, 1)].ravel()
    cs = cs[~np.isnan(cs)]
    # compute average distance between errs
    d = np.nanmean(average_distance(y_errs))
    return cs, d


def n_random_errs(n: int, num_errors: int) -> np.ndarray:
    """
    Parameters
    ----------
    n: int
        Size of possible total error set

    num_errors: int
        Number of y_err to produce

    Notes
    -----
    As EC is an IoU, if some number of samples are always correct, they don't
    factor into EC calculations. So the EC is calculated only based on the
    total error set size (set of samples which will at least once be classified
    incorrectly).
    """
    rng: Generator = np.random.default_rng()
    y_errs = []
    for i in range(num_errors):
        k = rng.integers(0, n + 1)
        errs = np.concatenate([np.ones(k, dtype=bool), np.zeros(n - k, dtype=bool)])
        y_errs.append(rng.permutation(errs))
    matrix = error_consistencies_numba(np.array(y_errs), empty_unions="nan")
    cs = matrix[np.triu_indices_from(matrix, 1)].ravel()
    cs = cs[~np.isnan(cs)]
    return cs


def n_random_k_max_errs(n: int, k_max: int, num_errors: int) -> np.ndarray:
    """
    Parameters
    ----------
    n: int
        Size of possible total error set

    k_max: int
        Number of errors < n produced each time (constant across repeats)

    num_errors: int
        Number of y_err to produce

    incremental: bool = False
        If True, log EC every 10 new error sets, to help determine repeats
        needed for convergence.

    Notes
    -----
    As EC is an IoU, if some number of samples are always correct, they don't
    factor into EC calculations. So the EC is calculated only based on the
    total error set size (set of samples which will at least once be classified
    incorrectly).
    """
    rng: Generator = np.random.default_rng()
    y_errs = []
    for i in range(num_errors):
        k = rng.integers(0, k_max + 1)
        errs = np.concatenate([np.ones(k, dtype=bool), np.zeros(n - k, dtype=bool)])
        y_errs.append(rng.permutation(errs))
    matrix = error_consistencies_numba(np.array(y_errs), empty_unions="nan")
    cs = matrix[np.triu_indices_from(matrix, 1)].ravel()
    cs = cs[~np.isnan(cs)]
    return cs


def n_fixed_k_errs(n: int, k: int, num_errors: int, incremental: bool = False) -> np.ndarray:
    """
    Parameters
    ----------
    n: int
        Size of possible total error set

    k: int
        Number of errors < n produced each time (constant across repeats)

    num_errors: int
        Number of y_err to produce

    incremental: bool = False
        If True, log EC every 10 new error sets, to help determine repeats
        needed for convergence.

    Notes
    -----
    As EC is an IoU, if some number of samples are always correct, they don't
    factor into EC calculations. So the EC is calculated only based on the
    total error set size (set of samples which will at least once be classified
    incorrectly).
    """
    rng: Generator = np.random.default_rng()
    errs = np.concatenate([np.ones(k, dtype=bool), np.zeros(n - k, dtype=bool)])
    if incremental:
        y_errs = []
        desc = f"(n={n}, k={k}) Mean EC: {{:0.3f}}"
        pbar = tqdm(range(num_errors), desc=desc.format(0), leave=False)
        for i in pbar:
            y_errs.append(rng.permutation(errs))
            if i % (num_errors // 10) == 0 and i > 0:
                matrix = error_consistencies_numba(np.array(y_errs), empty_unions="nan")
                cs = matrix[np.triu_indices_from(matrix, 1)].ravel()
                cs = cs[~np.isnan(cs)]
                pbar.set_description(desc.format(np.mean(cs)))
        matrix = error_consistencies_numba(np.array(y_errs), empty_unions="nan")
        cs = matrix[np.triu_indices_from(matrix, 1)].ravel()
        cs = cs[~np.isnan(cs)]
        return cs
    else:
        y_errs = []
        for i in range(num_errors):
            y_errs.append(rng.permutation(errs))
        matrix = error_consistencies_numba(np.array(y_errs), empty_unions="nan")
        cs = matrix[np.triu_indices_from(matrix, 1)].ravel()
        cs = cs[~np.isnan(cs)]
        return cs


def get_n_fixed_k_info(args: Namespace) -> Optional[DataFrame]:
    n, k = args.n, args.k
    ecs = n_fixed_k_errs(n, k, args.num_samples, incremental=False)
    mn, p5, p25, med, p75, p95, mx = np.percentile(ecs, [0, 5, 25, 50, 75, 95, 100])
    return pd.DataFrame(
        {
            "n": n,
            "k": k,
            "mean EC": np.mean(ecs),
            "sd EC": np.std(ecs, ddof=1),
            "min EC": mn,
            "5th pctile": p5,
            "25th pctile": p25,
            "median": med,
            "75th pctile": p75,
            "95th pctile": p95,
            "max EC": mx,
        },
        index=[0],
    )


def get_n_random_k_max_info(args: Namespace) -> Optional[DataFrame]:
    n, k_max = args.n, args.k_max
    ecs = n_fixed_k_errs(n, k_max, args.num_samples, incremental=False)
    mn, p5, p25, med, p75, p95, mx = np.percentile(ecs, [0, 5, 25, 50, 75, 95, 100])
    return pd.DataFrame(
        {
            "n": n,
            "k_max": k_max,
            "mean EC": np.mean(ecs),
            "sd EC": np.std(ecs, ddof=1),
            "min EC": mn,
            "5th pctile": p5,
            "25th pctile": p25,
            "median": med,
            "75th pctile": p75,
            "95th pctile": p95,
            "max EC": mx,
        },
        index=[0],
    )


def get_n_random_info(args: Namespace) -> Optional[DataFrame]:
    n = args.n
    ecs = n_random_errs(n, args.num_samples)
    mn, p5, p25, med, p75, p95, mx = np.percentile(ecs, [0, 5, 25, 50, 75, 95, 100])
    return pd.DataFrame(
        {
            "n": n,
            "mean EC": np.mean(ecs),
            "sd EC": np.std(ecs, ddof=1),
            "min EC": mn,
            "5th pctile": p5,
            "25th pctile": p25,
            "median": med,
            "75th pctile": p75,
            "95th pctile": p95,
            "max EC": mx,
        },
        index=[0],
    )


def get_n_correlated_info(args: Namespace) -> Optional[DataFrame]:
    n, gamma = args.n, args.gamma
    ecs, d = n_correlated_errs(n, gamma, args.num_samples)
    mn, p5, p25, med, p75, p95, mx = np.percentile(ecs, [0, 5, 25, 50, 75, 95, 100])
    return pd.DataFrame(
        {
            "n": n,
            "gamma": gamma,
            "similarity": 1 - d,
            "mean EC": np.mean(ecs),
            "sd EC": np.std(ecs, ddof=1),
            "min EC": mn,
            "5th pctile": p5,
            "25th pctile": p25,
            "median": med,
            "75th pctile": p75,
            "95th pctile": p95,
            "max EC": mx,
        },
        index=[0],
    )


def test_n_fixed_k_errs(capsys: CaptureFixture) -> None:
    ns = [5, 10, 25, 50, 100, 250, 500, 1000]
    ks = []
    for n in ns:
        ks.extend([int(perc * n) for perc in [0.2, 0.4, 0.6, 0.8, 0.9]])
    ks = np.unique(ks)
    with capsys.disabled():
        grid = list(ParameterGrid(dict(n=ns, k=ks, num_samples=[NUM_SAMPLES])))
        args = [Namespace(**g) for g in grid if g["n"] >= g["k"]]
        dfs = process_map(get_n_fixed_k_info, args)
        df = pd.concat(dfs, axis=0, ignore_index=True).sort_values(by=["n", "k"])
        outfile = OUTDIR / f"n_fixed_k_{NUM_SAMPLES}samples.csv"
        df.to_csv(outfile)
        print(f"Saved results to {outfile}")
        pd.options.display.max_rows = 999
        print(df)


def test_n_random_k_max_errs(capsys: CaptureFixture) -> None:
    ns = [5, 10, 25, 50, 100, 250, 500, 1000]
    k_max = []
    for n in ns:
        k_max.extend([int(perc * n) for perc in [0.2, 0.4, 0.6, 0.8, 0.9]])
    k_max = np.unique(k_max)
    with capsys.disabled():
        grid = list(ParameterGrid(dict(n=ns, k_max=k_max, num_samples=[NUM_SAMPLES])))
        args = [Namespace(**g) for g in grid if g["n"] >= g["k_max"]]
        dfs = process_map(get_n_random_k_max_info, args)
        df = pd.concat(dfs, axis=0, ignore_index=True).sort_values(by=["n", "k_max"])
        outfile = OUTDIR / f"n_random_k_max_{NUM_SAMPLES}samples.csv"
        df.to_csv(outfile)
        print(f"Saved results to {outfile}")
        pd.options.display.max_rows = 999
        print(df)


def test_n_random_errs(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        grid = list(
            ParameterGrid(
                dict(
                    n=[5, 10, 25, 50, 100, 250, 500, 1000, 5000],
                    num_samples=[2 * NUM_SAMPLES],
                )
            )
        )
        args = [Namespace(**g) for g in grid]
        dfs = process_map(get_n_random_info, args, max_workers=4)
        df = pd.concat(dfs, axis=0, ignore_index=True)
        outfile = OUTDIR / f"n_random_{2*NUM_SAMPLES}samples.csv"
        df.to_csv(outfile)
        print(f"Saved results to {outfile}")
        pd.options.display.max_rows = 999
        print(df)


def test_n_correlated_errs(capsys: CaptureFixture) -> None:
    NUM_SAMPLES = 1000
    with capsys.disabled():
        grid = list(
            ParameterGrid(
                dict(
                    n=[5, 10, 25, 50, 100, 250, 500, 1000, 5000],
                    gamma=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    num_samples=[NUM_SAMPLES],
                )
            )
        )
        args = [Namespace(**g) for g in grid]
        dfs = process_map(get_n_correlated_info, args, max_workers=4)
        df = pd.concat(dfs, axis=0, ignore_index=True).sort_values(by=["n", "gamma", "similarity"])
        outfile = OUTDIR / f"n_correlated_{NUM_SAMPLES}samples.csv"
        df.to_csv(outfile)
        print(f"Saved results to {outfile}")
        pd.options.display.max_rows = 999
        print(df)
