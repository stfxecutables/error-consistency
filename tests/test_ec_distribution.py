import timeit
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from _pytest.capture import CaptureFixture
from matplotlib import ticker
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numba import jit, prange
from numpy import ndarray
from numpy.random import Generator
from pandas import DataFrame, Series
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from error_consistency.functional import error_consistencies, error_consistencies_numba, get_y_error

OUTDIR = Path(__file__).resolve().parent / "test_results"
NUM_SAMPLES = 10000
OUTPUTS = dict(
    n_random_k=OUTDIR / "n_random_k_max_10000samples.csv",
    n_fixed_k=OUTDIR / "n_fixed_k_10000samples.csv",
    perturbed=OUTDIR / "n_perturbed_k_1000samples.csv",
    binomial=OUTDIR / "n_binomial_1000samples.csv",
    correlated=OUTDIR / "n_correlated_1000samples.csv",
)


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


def n_binomial_errs(n: int, p: float, num_errors: int) -> Tuple[ndarray, ndarray]:
    rng: Generator = np.random.default_rng()
    y_errs = []
    y_errs = rng.binomial(1, p, [num_errors, n])
    # for i in range(num_errors):
    #     k = rng.integers(0, n + 1)
    #     errs = np.concatenate([np.ones(k, dtype=bool), np.zeros(n - k, dtype=bool)])
    #     y_errs.append(rng.choice(errs, size=n, p=p, replace=True))
    # y_errs = np.array(y_errs)
    matrix = error_consistencies_numba(y_errs, empty_unions="nan")
    cs = matrix[np.triu_indices_from(matrix, 1)].ravel()
    cs = cs[~np.isnan(cs)]
    # compute average distance between errs
    d = np.nanmean(average_distance(y_errs))
    return cs, d


def n_perturbed_k_errs(
    n: int, k_base: int, noise: float, num_errors: int
) -> Tuple[ndarray, ndarray]:
    """Generate correlated errors by adding a 'noise' vector to a base error vector. Basically,
    a XOR 1 flips a bit. (0 ^ 1) = 1, (1 ^ 1) = 0. So given an error set `e`, and noise vector `noise`,
    where noise is sampled from a binomial with noise `p`, `e ^ noise` gives us a new perturbed error
    vector with the amount of bits flipped depending on `p`. As `p` approaches 1, we flip all bits, so
    all errors become the same, so this is actually max correlation. Same with `p` == 0. So we really
    need to vary `p` between 0 and 0.5, where 0.5 is maximum noise.

    n: int
        Total error set size

    k_base: int
        Number of base errors to start with.

    p: int
        Noise value for Bernoulli noise.
    """
    rng: Generator = np.random.default_rng()
    errs = np.concatenate([np.ones(k_base, dtype=bool), np.zeros(n - k_base, dtype=bool)])
    y_errs = []
    for i in range(num_errors):
        noise_vec = rng.binomial(1, noise, size=n)
        y_errs.append(errs ^ noise_vec)  # use XOR so we aren't just increasing amount of 1s
    matrix = error_consistencies_numba(np.array(y_errs), empty_unions="nan")
    cs = matrix[np.triu_indices_from(matrix, 1)].ravel()
    cs = cs[~np.isnan(cs)]
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


def get_n_binomial_info(args: Namespace) -> Optional[DataFrame]:
    n, p = args.n, args.p
    ecs, d = n_binomial_errs(n, p, args.num_samples)
    mn, p5, p25, med, p75, p95, mx = np.percentile(ecs, [0, 5, 25, 50, 75, 95, 100])
    return pd.DataFrame(
        {
            "n": n,
            "p": p,
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


def get_n_perturbed_k_info(args: Namespace) -> Optional[DataFrame]:
    n, k_base, noise = args.n, args.k_base, args.noise
    ecs, d = n_perturbed_k_errs(n, k_base, noise, args.num_samples)
    mn, p5, p25, med, p75, p95, mx = np.percentile(ecs, [0, 5, 25, 50, 75, 95, 100])
    return pd.DataFrame(
        {
            "n": n,
            "k_base": k_base,
            "noise": noise,
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


def test_n_binomial_errs(capsys: CaptureFixture) -> None:
    NUM_SAMPLES = 1000
    with capsys.disabled():
        grid = list(
            ParameterGrid(
                dict(
                    n=[5, 10, 25, 50, 100, 250, 500, 1000, 5000],
                    p=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    num_samples=[NUM_SAMPLES],
                )
            )
        )
        args = [Namespace(**g) for g in grid]
        dfs = process_map(get_n_binomial_info, args, max_workers=8)
        df = pd.concat(dfs, axis=0, ignore_index=True).sort_values(by=["n", "p", "similarity"])
        outfile = OUTDIR / f"n_binomial_{NUM_SAMPLES}samples.csv"
        df.to_csv(outfile)
        print(f"Saved results to {outfile}")
        pd.options.display.max_rows = 999
        print(df)


def test_n_perturbed_errs(capsys: CaptureFixture) -> None:
    NUM_SAMPLES = 1000
    ns = [5, 10, 25, 50, 100, 250, 500, 1000]
    ks = []
    for n in ns:
        ks.extend([int(perc * n) for perc in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]])
    ks = np.unique(ks)
    with capsys.disabled():
        grid = list(
            ParameterGrid(
                dict(
                    n=ns,
                    k_base=ks,
                    noise=np.arange(0.05, 0.55, 0.05),
                    num_samples=[NUM_SAMPLES],
                )
            )
        )
        args = [Namespace(**g) for g in grid if g["k_base"] < g["n"]]
        dfs = process_map(get_n_perturbed_k_info, args, max_workers=8)
        df = pd.concat(dfs, axis=0, ignore_index=True).sort_values(
            by=["n", "k_base", "noise", "similarity"]
        )
        outfile = OUTDIR / f"n_perturbed_k_{NUM_SAMPLES}samples.csv"
        df.to_csv(outfile)
        print(f"Saved results to {outfile}")
        pd.options.display.max_rows = 9999
        print(df)


def best_rect(m: int) -> Tuple[int, int]:
    """returns dimensions (smaller, larger) of closest rectangle"""
    low = int(np.floor(np.sqrt(m)))
    high = int(np.ceil(np.sqrt(m)))
    prods = [(low, low), (low, low + 1), (high, high), (high, high + 1)]
    for i, prod in enumerate(prods):
        if prod[0] * prod[1] >= m:
            return prod
    raise ValueError("Unreachable!")


def plot_k_tables(use_3d: bool = False, show: bool = False) -> None:
    df_k = pd.read_csv(OUTPUTS["n_fixed_k"]).iloc[:, 1:]  # ignore stupid csv index
    df_kmax = pd.read_csv(OUTPUTS["n_random_k"]).iloc[:, 1:]
    if use_3d:
        # just plot x =n, y = k, ECs, percentiles = colors
        sbn.set_style("white")
        palette = sbn.color_palette("Spectral", as_cmap=True)
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection="3d", azim=-26, elev=15)
        ax2 = fig.add_subplot(1, 2, 2, projection="3d", azim=-128, elev=3)
        x, y, z = df_k["n"], df_k["k"], df_k["mean EC"]
        sd = 100 * df_k["sd EC"]

        for ax in [ax1, ax2]:
            ax.scatter3D(x, y, z, s=sd, cmap=palette)
            ax.set_xlabel("n")
            ax.set_ylabel("k")
            ax.set_zlabel("mean EC")
        fig.set_size_inches(w=9, h=4)
        fig.suptitle("AEC with Total Error set size n\n fixed size k error sets")
        fig.subplots_adjust(top=1.0, bottom=0.0, left=0.05, right=0.95, hspace=0.2, wspace=0.3)
        if show:
            plt.show()
        else:
            fig.savefig(OUTDIR / f"n_k_AECs_3d.png", dpi=300)
            plt.close()
        return
    sbn.set_style("darkgrid")
    # palette = sbn.color_palette("mako", as_cmap=True)
    # palette = sbn.color_palette("rocket", as_cmap=True)
    # palette = sbn.color_palette("Spectral", as_cmap=True)
    palette = sbn.dark_palette("#0084ff", as_cmap=True)
    fig, axes = plt.subplots(ncols=2)

    # plot fixed k
    x, k, y, sd = df_k["n"], df_k["k"], df_k["mean EC"], df_k["sd EC"]
    k /= x
    k.name = "k/n"
    y.name = "AEC"
    sbn.scatterplot(
        x=x,
        y=y,
        hue=k,
        size=sd,
        ax=axes[0],
        palette=palette,
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("EC with Total Error set size n and fixed k random errors")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("EC")
    axes[0].yaxis.set_major_formatter(ticker.ScalarFormatter())
    axes[0].xaxis.set_major_formatter(ticker.ScalarFormatter())

    # plot random k_max
    x, k, y, sd = df_kmax["n"], df_kmax["k_max"], df_kmax["mean EC"], df_kmax["sd EC"]
    k /= x
    k.name = "k_max/n"
    y.name = "AEC"
    sbn.scatterplot(
        x=x,
        y=y,
        hue=k,
        size=sd,
        ax=axes[1],
        palette=palette,
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("EC with Total Error set size n and up to k random errors")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("EC")
    axes[1].yaxis.set_major_formatter(ticker.ScalarFormatter())
    axes[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
    fig.set_size_inches(w=12, h=6)
    if show:
        plt.show()
    else:
        fig.savefig(OUTDIR / f"n_k_AECs.png", dpi=300)
        plt.close()
    return


def plot_correlated_tables(use_3d: bool = False, show: bool = True) -> None:
    perturbed = pd.read_csv(OUTPUTS["perturbed"]).iloc[:, 1:]  # ignore stupid csv index
    if use_3d:
        # just plot x =n, y = k, ECs, percentiles = colors
        sbn.set_style("white")
        # palette = sbn.color_palette("Spectral", as_cmap=True)
        palette = sbn.color_palette("icefire", as_cmap=True)
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection="3d", azim=-163, elev=11)
        ax2 = fig.add_subplot(1, 2, 2, projection="3d", azim=-66, elev=11)
        df = perturbed
        x = df["k_base"] / df["n"]
        z = df["mean EC"]
        x.name = "k_base / n"
        z.name = "AEC"
        y = df["noise"]
        for ax in [ax1, ax2]:
            # ax.scatter3D(x, y, z, s=100 * df["sd EC"], color="#004ca3")
            ax.plot_trisurf(x, y, z, cmap=palette, lw=0.1)
            ax.set_xlabel(x.name)
            ax.set_ylabel(y.name)
            ax.set_zlabel(z.name)
        fig.set_size_inches(w=9, h=4)
        fig.suptitle(
            "AEC with Total Error set size n\nerror sets perturbed from base set of fixed k_base errors"
        )
        fig.subplots_adjust(top=1.0, bottom=0.0, left=0.05, right=0.95, hspace=0.2, wspace=0.026)
        if show:
            plt.show()
        else:
            fig.savefig(OUTDIR / f"perturbation_3d.png", dpi=300)
            plt.close()
        return

    sbn.set_style("darkgrid")
    # palette = sbn.dark_palette("#ff5900", as_cmap=True)
    palette = sbn.color_palette("icefire", as_cmap=True)
    fig, ax = plt.subplots()

    # plot perturbed
    # remove erroneous choices of noise
    # perturbed = perturbed.loc[perturbed["noise"] <= 0.8]
    # perturbed["noise"] = perturbed["noise"].apply(lambda x: 1 - x if x > 0.5 else x)
    n, k_base, y, sd = perturbed["n"], perturbed["k_base"], perturbed["mean EC"], perturbed["sd EC"]
    x = k_base / n
    x.name = "k_base / n"
    y.name = "AEC"
    noise = perturbed["noise"]
    sbn.scatterplot(x=x, y=y, hue=noise, size=k_base, ax=ax, palette=palette, alpha=0.7)
    ax.set_title(
        "AEC with Total Error set size n\nerror sets perturbed from base set of fixed k_base errors"
    )
    ax.set_xlabel("k_base / n")
    ax.set_ylabel("AEC")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    # axes[0].yaxis.set_minor_formatter(None)
    # axes[0].xaxis.set_minor_formatter(None)
    fig.legend(*ax.get_legend_handles_labels())
    ax.get_legend().remove()
    fig.set_size_inches(w=12, h=7)
    if show:
        plt.show()
    else:
        fig.savefig(OUTDIR / f"perturbation_2d.png", dpi=300)
        plt.close()
    return


def test_plot_results(capsys: CaptureFixture) -> None:
    # plot_k_tables(use_3d=False, show=False)
    # plot_k_tables(use_3d=True, show=False)
    plot_correlated_tables(use_3d=True, show=False)
    plot_correlated_tables(use_3d=False, show=False)
