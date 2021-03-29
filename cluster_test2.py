import sys
from itertools import repeat
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from numpy import ndarray
from pandas import DataFrame
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from error_consistency.functional import error_consistencies_numba


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


def bootstrap_cod(subsample: List[ndarray]) -> DataFrame:
    COLUMNS = ["n_clusterings", "disagree_pairs", "mean_cod", "sd_cod"]
    disagreements = []
    for i, clust1 in enumerate(subsample):
        for j, clust2 in enumerate(subsample, start=i + 1):
            c1, c2 = np.copy(clust1), np.copy(clust2)
            c2 = hungarian_remap(c1, c2)
            disagreements.append(c1 != c2)

    cod_array = error_consistencies_numba(np.array(disagreements), empty_unions="1")
    cods = cod_array[np.triu_indices_from(cod_array, k=1)].ravel()
    mean, sd = np.mean(cods), np.std(cods, ddof=1)
    return pd.DataFrame(columns=COLUMNS, data=[[i, len(cods), mean, sd]])


def get_cluster(args: Dict) -> ndarray:
    """kwargs should be (k: int, X: ndarray, y: ndarray)"""
    k, X, y = args["k"], args["X"], args["y"]
    return KMeans(k, max_iter=1).fit_predict(X, y)


def generate_clusterings(
    n_clusterings: int, n_samples: int = 100, k: int = 2, preview: bool = False, n_workers: int = 2
) -> List[ndarray]:
    # generate random data to perform clustering on
    N = np.random.randint(4, 7)
    data = random_gaussians(n_gaussians=N, n_samples=n_samples)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    if preview:
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1])
        plt.ioff()
        plt.show()
        cont = input("Do you wish to use this data for clustering? [Y/n]\n")
        if cont not in ["Y", "y", ""]:
            print("Aborting.")
            sys.exit()
        else:
            plt.close()

    print(f"Computing for K = {k} in K-means...")
    args = dict(k=k, X=X, y=y)
    clusterings: List[ndarray] = process_map(
        get_cluster,
        repeat(args, n_clusterings),
        chunksize=2,
        max_workers=n_workers,
        total=n_clusterings,
        desc=f"{'Generating clusterings':.<30}",
    )
    # clusterings = []
    # for _ in tqdm(
    #     range(n_clusterings), total=n_clusterings, desc=f"{'Generating clusterings':.<30}"
    # ):
    #     clusterings.append(KMeans(k, max_iter=50).fit_predict(X, y))
    return clusterings


def compute_bootstrap_cods(
    clusterings: List[ndarray], n_subsamples: int, n_workers: int = 2
) -> DataFrame:
    N_CLUSTERINGS = len(clusterings)
    MIN_SIZE = np.max([3, N_CLUSTERINGS // 2])
    dfs = []
    subsamples = []
    for _ in range(n_subsamples):
        size = np.random.randint(MIN_SIZE, N_CLUSTERINGS, dtype=int)  # need three clusters min
        idx = list(range(N_CLUSTERINGS))
        idx_boot = np.random.choice(idx, size, replace=True)
        subsample = np.array(clusterings)[idx_boot]
        subsamples.append(subsample)
    subsamples.append(clusterings)  # always ensure we measure the full set at least once

    desc = f"{'Computing boot CoDs':.<30}"
    if n_workers > 1:
        dfs = process_map(  # parallelize bootstrapping
            bootstrap_cod,
            subsamples,
            max_workers=2,  # leave some cores for consistency parallelization?
            chunksize=20 if len(clusterings[0]) > 1000 else None,
            total=n_subsamples,
            desc=f"{'Computing boot CoDs':.<30}",
            leave=True,
        )
        return pd.concat(dfs)
    dfs = []
    for subsample in tqdm(subsamples, total=n_subsamples, desc=desc):
        dfs.append(bootstrap_cod(subsample))
    return pd.concat(dfs)


def plot_bootstrap_results(df: DataFrame) -> None:
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(ncols=2)
    axes[0].scatter(df["disagree_pairs"], df["mean_cod"], label="mean", color="black", s=2.0)
    axes[0].scatter(df["disagree_pairs"], df["sd_cod"], label="sd", color="red", s=2.0)
    axes[0].legend().set_visible(True)
    axes[0].set_xlabel("Number of Disagreement-Pairings Included")

    axes[1].scatter(df["n_clusterings"], df["mean_cod"], label="mean", color="black", s=2.0)
    axes[1].scatter(df["n_clusterings"], df["sd_cod"], label="sd", color="red", s=2.0)
    axes[1].legend().set_visible(True)
    axes[1].set_xlabel("Number of Clusterings")

    final_mean = np.round(np.mean(df["mean_cod"]), 2)
    final_sd = np.round(np.mean(df["sd_cod"]), 3)
    fig.suptitle(
        f"Estimated Convergence Rate of Consistency of Disagreement\nFinal: mean={final_mean}, sd={final_sd}"
    )
    fig.set_size_inches(w=8, h=5)
    plt.show()


if __name__ == "__main__":
    # fmt: off
    N_SAMPLES = 10000      # number of datapoints to cluster
    N_CLUSTERINGS = 100  # number of times to repeatedly cluster the datapoints
    N_SUBSAMPLES = 50   # bootstrap resampling clusterings to estimate convergence
    # fmt: on
    # if N_CLUSTERINGS == 100:
    #     np.random.seed(6)
    # elif N_CLUSTERINGS == 200:
    #     np.random.seed(3)
    K = 2  # number of K-Means clusters

    clusterings = generate_clusterings(N_CLUSTERINGS, N_SAMPLES, k=K, preview=True)
    df = compute_bootstrap_cods(clusterings, N_SUBSAMPLES, n_workers=1)
    plot_bootstrap_results(df)
