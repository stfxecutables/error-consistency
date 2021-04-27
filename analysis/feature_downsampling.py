import os
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing import cast, no_type_check
from typing_extensions import Literal
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from argparse import ArgumentParser, Namespace
from numpy import ndarray
from pandas import DataFrame, Series
from enum import Enum
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from warnings import filterwarnings

sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis.constants import (
    PERCENT_MAX,
    PERCENT_MIN,
    DOWNSAMPLE_PLOT_OUTDIR,
    DOWNSAMPLE_RESULTS_DIR,
    N_ROWS,
    REPS_PER_PERCENT,
    KFOLD_REPS,
    PERCENTS,
    COLS,
    CLASSIFIERS,
    DATA,
)
from analysis.downsampling_setup import argparse_setup, classifier_from_args, dataset_from_args
from error_consistency.consistency import (
    ErrorConsistencyKFoldHoldout,
    ErrorConsistencyKFoldInternal,
)


CLASSIFIER_CHOICES = ["knn1", "knn3", "knn5", "knn10", "lr", "svm", "rf", "ada", "mlp"]
DATASET_CHOICES = ["diabetes", "park", "trans", "spect"]
SCRIPT_OUTDIR = Path(__file__).resolve().parent.parent

FlatArray = Union[DataFrame, Series, ndarray]


def overall_cohen_d(X: DataFrame, y: FlatArray, statistic: str = "mean") -> float:
    """For each feature `x` in `X`, compute the absolute [see notes] Cohen's d value and return a
    summary statistic of those values.

    Parameters
    ----------
    X: DataFrame
        DataFrame with shape (n_samples, n_features), and no target variable included

    y: DataFrame | Series | ndarray
        DataFrame, Series, or ndarray with shape (n_samples,) and only two unique values (0, 1).

    statistic: "mean" | "median"
        How to summarize the resulting rho values.

    Notes
    -----
    Because we are summarizing multiple Cohen's d values from different features, retaining the sign
    is invalid (e.g. each feature has its own unit of measure, and allowing "-1 f1" to cancel out `1
    f2" for units "f1" and "f2" is invalid.
    """
    data = X.to_numpy()
    x1, x2 = data[y == 0], data[y == 1]
    n1, n2 = len(x1) - 1, len(x2) - 1
    sd1, sd2 = np.std(x1, ddof=1, axis=0), np.std(x2, ddof=1, axis=0)
    sd_pools = np.sqrt((n1 * sd1 + n2 * sd2) / (n1 + n2))
    ds = np.abs(np.mean(x1, axis=0) - np.mean(x2, axis=0)) / sd_pools
    summary = np.mean if statistic == "mean" else np.median
    return float(summary(ds.ravel()))


def overall_auroc(X: DataFrame, y: FlatArray, statistic: str = "mean") -> float:
    """For each feature `x` in `X`, compute rho, the common-language effect size (see Notes) via the
    area-under-the-ROC curve (AUC) using `x` as the `y_score`, rescale this effect size to allow
    aggregation across features, and return a summary statistic of those rescaled values.

    Parameters
    ----------
    X: DataFrame
        DataFrame with shape (n_samples, n_features), and no target variable included

    y: DataFrame | Series | ndarray
        DataFrame, Series, or ndarray with shape (n_samples,) and only two unique values (0, 1).

    statistic: "mean" | "median"
        How to summarize the resulting rho values.

    Notes
    -----
    This is equivalent to calculating U / abs(y1*y2), where `y1` and `y2` are the subgroup sizes,
    and U is the Mann-Whitney U, and is also sometimes referred to as Herrnstein's rho [1] or "f",
    the "common-language effect size" [2].

    Note we also *must* rescale due to the implicit different units of each feature. E.g. as rho
    is implicity "signed", with values above 0.5 indicating separation in one
    direction, and values below 0.5 indicating separation in the other, taking a mean or median can
    be deeply misleading if some features result in rho values close to 1, and others have rho
    values close to zero.

    [1] Herrnstein, R. J., Loveland, D. H., & Cable, C. (1976). Natural concepts in pigeons.
        Journal of Experimental Psychology: Animal Behavior Processes, 2, 285-302

    [2] McGraw, K.O.; Wong, J.J. (1992). "A common language effect size statistic". Psychological
        Bulletin. 111 (2): 361–365. doi:10.1037/0033-2909.111.2.361.
    """
    if statistic not in ["mean", "median"]:
        raise ValueError("Must use either 'mean' or 'median' for `statistic`.")
    aucs = [roc_auc_score(y, X[x]) for x in X.columns]
    aucs = [np.abs(auc - 0.5) for auc in aucs]
    summary = np.mean if statistic == "mean" else np.median
    return float(summary(np.array(aucs).ravel()))


def mahalanobi(X: DataFrame, y: FlatArray, center: str = "mean"):
    """

    Parameters
    ----------
    X: DataFrame
        DataFrame with shape (n_samples, n_features), and no target variable included

    y: DataFrame
        DataFrame, Series, or ndarray with shape (n_samples,) and only two unique values (0, 1).

    Notes
    -----
    Finding a measure to convey the distance of two subgroups given multiple features is essentially
    a search for a cluster similarity / distance metric. If we replace subgroups with their mean
    vectors (or median vectors), then we can use a number of natural multi-dimensional distance
    measures. However, only a few of these take into account variance and/or overlap. Using
    Mahalanobis distance on the subgroup means/medians has this partial advantage (naturally
    rescales the data). Otherwise, data could be feature-wise centered and then standard distance
    metrics used.
    """
    if center not in ["mean", "median"]:
        raise ValueError("Must use either 'mean' or 'median' for `center`.")
    pass


def separations(X: DataFrame, y: Union[DataFrame, Series, ndarray]) -> DataFrame:
    """Return the area under the ROC curve of a perfect classifier, Mahalanobis D, and average
    Cohen's d values from data `X` with **two** subgroups defined in `y`

    Parameters
    ----------
    X: DataFrame
        DataFrame with shape (n_samples, n_features), and no target variable included

    y: DataFrame
        DataFrame, Series, or ndarray with shape (n_samples,) and only two unique values (0, 1).

    Returns
    -------
    val1: Any
    """
