from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing import cast, no_type_check
from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.cluster import KMeans
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix

from error_consistency.functional import error_consistencies_numba

from numba import jit, prange


def random_gaussians(n_gaussians: int = 5, n_samples: int = 100):
    """Generate random clusters with multivariate normal distributions centered at various points
    in [-5, 5]^2 and with variances in [0, 2].
    """
    dfs = []
    for i in range(n_gaussians):
        mean = np.random.uniform(-5, 5, 2)
        cov = np.random.uniform(-2, 2, [2, 2]) + np.eye(2) / 10000
        cov = cov @ cov.T  # make positive definite
        x = np.random.multivariate_normal(mean, cov, n_samples)
        data = pd.DataFrame(columns=["x1", "x2", "y"], index=range(n_samples), dtype=float)
        data.x1 = x[:, 0]
        data.x2 = x[:, 1]
        data.y = i
        dfs.append(data)
    return pd.concat(dfs, axis=0)


def hungarian_remap(labels1: ndarray, labels2: ndarray) -> ndarray:
    """Remap the labels in `labels2` to the labels in `labels1` in a way that maximizes overlap.
    Assumes all labels start at 0. Also assumes the number of unique labels in `labels1` and
    `labels2` are the same (same number of clusters for each).
    """
    C = contingency_matrix(labels1, labels2)
    row_idx, col_idx = linear_sum_assignment(C, maximize=True)
    new_labels2 = np.zeros_like(labels2) - 1
    for i, j in zip(row_idx, col_idx):
        new_labels2[labels2 == j] = i
    return new_labels2


def remap_from_matrix(labels1: ndarray, labels2: ndarray, contingencies: ndarray) -> Any:
    """Remap `labels2` to `labels` using contingency matrix `contingencies`.

    Returns
    -------
    two_to_one: ndarray
        `labels2` remapped to `labels1`

    one_to_two: ndarray
        `labels1` remapped to `labels2`
    """
    row_idx, col_idx = linear_sum_assignment(contingencies, maximize=True)
    # if one of `matches` below is equal to (i, j), this means clustering label i + 1 from `labels1`
    # aligns best with clustering label j + 1 from `labels2`
    two_to_one = np.full_like(labels2, -1, dtype=int)
    one_to_two = np.full_like(labels1, -1, dtype=int)
    for i, j in zip(row_idx, col_idx):
        if contingencies[i, j] == 0:
            continue  # do not remap padded / zero interection cells
        two_to_one[labels2 == j] = i
        one_to_two[labels1 == i] = j

    return two_to_one, one_to_two


def unbalanced_hungarian_remap(labels1: ndarray, labels2: ndarray) -> Tuple[ndarray, ...]:
    """Remap the labels in two clustering to the coarser or finer clustering. Assumes all labels
    start at 0, i.e. there are no zero labels.

    Python's `linear_sum_assignment` can actually *already* handle non-square matrices, and so we
    don't even need to zero-pad. In fact, it is possible zero-padding *may cause errors*, so perhaps
    we should NOT use `pad_to_square`.

    Parameters
    ----------
    labels1: ndarray
        Labels for the first clustering.

    labels2: ndarray
        Labels for the second clustering.

    to_less: bool = True
        If True, do "more-to-less" alignment, i.e. align the clustering with more labels to the
        clustering with less labels. Otherwise, do the reverse.

    Returns
    -------
    remapped1: ndarray
        The remapped labels for cluster 1. If the number of clusters in labels1 is less than or
        equal to the number of clusters in labels2, and `to_less=True`, then `remapped1 == labels1`.
        Otherwise, this equality may not hold.

    remapped2: ndarray
        The remapped labels for cluster 2. If the number of clusters in labels1 is greater than
        the number of clusters in labels2, and `to_less=False`, then `remapped1 == labels1`.
        Otherwise, this equality may not hold.
    """
    n_clust1, n_clust2 = len(np.unique(labels1)), len(np.unique(labels2))
    if n_clust1 == n_clust2:
        return hungarian_remap(labels1, labels2)
    C = contingency_matrix(labels1, labels2)
    two_to_one, one_to_two = remap_from_matrix(labels1, labels2, C)
    return two_to_one, one_to_two


@jit(nopython=True, fastmath=True)
def running_totals(cods: ndarray, n_steps: int) -> Tuple[ndarray, ndarray, ndarray]:
    C_TOTAL = len(cods)
    means = []
    sds = []
    steps = C_TOTAL // n_steps
    done = 0
    counted = []
    for i in range(0, C_TOTAL, steps):
        means.append(np.mean(cods[: i + 1]))
        counted.append(i)
        if i != 0:
            sds.append(np.std(cods[: i + 1]))
        if done % (n_steps // 20) == 0:
            print("Percent done: ", 100 * i / C_TOTAL)
        done += 1
    return means, sds, counted


def running_totals_decimated(cods: ndarray, decimation: int) -> Tuple[ndarray, ndarray, ndarray]:
    decimated = cods[::decimation]
    return running_totals(decimated, 1)


@jit(nopython=True, fastmath=True, parallel=True)
def running_totals_parallel(cods: ndarray, starts: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    N = len(starts)
    means = np.zeros((N,))
    sds = np.zeros((N - 1,))
    for i in prange(N):
        start = starts[i]
        means[i] = np.mean(cods[: start + 1])
        if i != 0:
            sds[i] = np.std(cods[: start + 1])
    return means, sds, starts

@jit(nopython=True, fastmath=True, parallel=True)
def running_totals_rand_parallel(cods: ndarray, n_samples: ndarray, size: int) -> Tuple[ndarray, ndarray, ndarray]:
    # This doesn't work, need to sample random rows/columns to match actual convergence
    means = np.zeros((n_samples,))
    sds = np.zeros((n_samples,))
    samples = np.zeros((n_samples * size))
    for i in prange(n_samples):
        sample = np.random.choice(cods, size)
        start = starts[i]
        means[i] = np.mean(cods[: start + 1])
        if i != 0:
            sds[i] = np.std(cods[: start + 1])
    return means, sds, starts


if __name__ == "__main__":
    N_CLUSTERINGS = 200
    if N_CLUSTERINGS == 100:
        np.random.seed(6)
    elif N_CLUSTERINGS == 200:
        np.random.seed(3)

    N = np.random.randint(4, 7)
    data = random_gaussians(n_gaussians=N)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1])
    plt.show()
    for k in range(2, 7):  # choice of `k` for KMeans
        print(f"Computing for K = {k} in K-means...")
        clusterings = []
        for _ in tqdm(range(N_CLUSTERINGS), total=N_CLUSTERINGS, desc="Clusterings"):
            clusterings.append(KMeans(k, max_iter=5).fit_predict(X, y))

        disagreements = []
        for i, clust1 in tqdm(enumerate(clusterings), total=N_CLUSTERINGS, desc="Disagreements"):
            for j, clust2 in enumerate(clusterings, start=i + 1):
                c1, c2 = np.copy(clust1), np.copy(clust2)
                c2 = hungarian_remap(c1, c2)
                disagreements.append(c1 != c2)

        # Below takes 13 minutes just for 100 clusterings, so is no good.
        """
        cods, means, sds = [], [], []
        for i, d1 in tqdm(enumerate(disagreements), total=len(disagreements), desc="Consistency"):
            for j, d2 in enumerate(disagreements, start=i + 1):
                union = np.sum(d1 | d2)
                if union <= 1.0:
                    cods.append(1.0)
                    continue
                denom = np.sum(d1 & d2)
                cods.append(union / denom)
                means.append(np.mean(cods))
                if len(cods) > 1:
                    sds.append(np.std(cods, ddof=1))
        """
        # Using my fast implementation instead this is still under a minute for 200 clusterings
        print("Computing fast consistencies...", end="", flush=True)
        cods_all = error_consistencies_numba(np.array(disagreements), empty_unions="1")
        print("done.")
        cods = cods_all[np.triu_indices_from(cods_all, k=1)].ravel()
        print(
            f"Mean C of Disagreement (sd): {np.mean(cods):0.3f} ({np.std(cods, ddof=1):0.3f})",
            flush=True,
        )
        C_TOTAL = len(cods)

        # again, below is way too slow
        """
        means, sds = [], []
        for i in tqdm(range(C_TOTAL), total=C_TOTAL, desc="Running totals"):
            means.append(cods[: i + 1])
            if i != 0:
                sds.append(cods[: i + 1])
        """

        # this is also way too slow:
        """
        print("Computing fast running totals...", end="", flush=True)
        means, sds = running_totals(cods, steps=1)
        print("done.")
        """

        # We instead approximate by only taking the mean every D steps
        # Even still this is extremely slow for just 200 clusterings
        # And results in strange shapes
        print("Computing fast running totals...\n")
        # means, sds, counted = running_totals(cods, n_steps=1000)
        C_TOTAL, n_steps = len(cods), 1000
        steps = C_TOTAL // n_steps
        starts = np.array(list(range(0, C_TOTAL, steps)), dtype=int)
        means, sds, counted = running_totals_parallel(cods, starts)

        # x = list(range(C_TOTAL))
        sbn.set_style("darkgrid")
        fig, axes = plt.subplots(ncols=2)
        # ax.plot(x, consistencies, label="consistency", color="black")
        axes[0].plot(counted[1:], means[1:], label="mean", color="black")
        axes[0].plot(counted[2:], sds[1:], label="sd")
        axes[0].legend().set_visible(True)
        axes[0].set_xlabel("Number of Disagreement-Pairings Included")

        required_k = (np.array(counted) * 8) ** 0.25
        axes[1].plot(required_k[1:], means[1:], label="mean", color="black")
        axes[1].plot(required_k[2:], sds[1:], label="sd")
        axes[1].legend().set_visible(True)
        axes[1].set_xlabel("Approximate Number of Clusterings Needed")

        final_mean = np.round(np.mean(means[-C_TOTAL // 10 :]), 2)
        final_sd = np.round(np.mean(sds[-C_TOTAL // 10 :]), 3)
        fig.suptitle(
            f"Estimated Convergence Rate of Consistency of Disagreement\nFinal: mean={final_mean}, sd={final_sd}"
        )
        fig.set_size_inches(w=8, h=5)
        plt.show()
