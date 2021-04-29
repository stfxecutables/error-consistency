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
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.append(str(Path(__file__).resolve().parent.parent))

from error_consistency.consistency import (
    ErrorConsistencyKFoldHoldout,
    ErrorConsistencyKFoldInternal,
)

CLASSIFIER_CHOICES = ["knn1", "knn3", "knn5", "knn10", "lr", "svm", "rf", "ada", "mlp"]
DATASET_CHOICES = ["diabetes", "park", "trans", "spect"]
JSON_DIR = Path(__file__).resolve().parent / "results/compute_canada/dfs"
PLOT_DIR = Path(__file__).resolve().parent / "results/compute_canada/plots"
PDF_DIR = PLOT_DIR / "pdf"
PNG_DIR = PLOT_DIR / "png"
if not PDF_DIR.exists():
    os.makedirs(PDF_DIR, exist_ok=True)
if not PNG_DIR.exists():
    os.makedirs(PNG_DIR, exist_ok=True)
JSONS = sorted(JSON_DIR.rglob("*.json"))
DATASETS = [file.name.split("_")[0] for file in JSONS]
DFS = [pd.read_json(json) for json in JSONS]


def plot_results(
    file: Path, df: DataFrame, jitter: bool = True, curve: bool = True, show: bool = False
) -> None:
    fig: plt.Figure
    ax: plt.Axes
    pieces = file.name.split("_")
    ds = dataset = pieces[0]
    classifier = file.name[len(ds) + 1 : file.name.find("__")]
    if classifier == "":
        return
    sbn.set_style("darkgrid")
    fig, ax = plt.subplots()
    x = df["Percent"].to_numpy().astype(float)
    acc = df["Accuracy"]
    con = df["Consistency"]
    if jitter:
        d = np.mean(np.diff(np.unique(x)))  # spacing for jitter
        x += np.random.uniform(0, d, len(x))  # x-jitter
    if curve:
        ex = df["Percent"].to_numpy()
        acc_smooth = lowess(acc, ex, return_sorted=False)
        con_smooth = lowess(con, ex, return_sorted=False)
        sbn.lineplot(x=ex, y=acc_smooth, color="black", label="Accuracy", alpha=0.6, lw=1, ax=ax)
        sbn.lineplot(x=ex, y=con_smooth, color="red", label="Consistency", alpha=0.6, lw=1, ax=ax)
    sbn.scatterplot(
        x=x, y=acc, color="black", label=None if curve else "Accuracy", alpha=0.6, s=3, ax=ax
    )
    sbn.scatterplot(
        x=x, y=con, color="red", label=None if curve else "Consistency", alpha=0.6, s=3, ax=ax
    )
    ax.set_title(f"{dataset} - {classifier}")
    ax.set_xlabel("Downsampling Percentage")
    ax.set_ylabel("Accuracy (or Consistency)")
    ax.set_xlim(50, 100)
    ax.set_ylim(0.2, 1)
    fig.set_size_inches(w=6, h=4)
    if show:
        plt.show()
        return
    j = "" if jitter else "_no-jitter"
    pdf = PDF_DIR / f"{file.stem}{j}.pdf"
    png = PNG_DIR / f"{file.stem}{j}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close()


if __name__ == "__main__":
    for file, df in tqdm(zip(JSONS, DFS), total=len(DFS), desc="Plotting"):
        if "MLP" not in file.name:
            continue
        plot_results(file, df, jitter=False)
    print(f"Saved plots to {PDF_DIR.parent}")
    # FILE = Path("/home/derek/Desktop/error-consistency/analysis/results/test_results/Diabetes_Logistic_Regression__k-fold-holdout_downsample.json")
    # df = pd.read_json(FILE)
    # plot_results(FILE, df, jitter=False, curve=True, show=True)
