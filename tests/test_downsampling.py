import os
from typing import Any, Dict, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from numpy import ndarray
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm

from error_consistency.consistency import (
    ErrorConsistencyKFoldHoldout,
    ErrorConsistencyKFoldInternal,
)
from error_consistency.testing.loading import CLASSIFIERS, DATA, OUTDIR

PLOT_OUTDIR = OUTDIR / "plots"
RESULTS_DIR = OUTDIR / "dfs"
if not PLOT_OUTDIR.exists():
    os.makedirs(PLOT_OUTDIR)
if not RESULTS_DIR.exists():
    os.makedirs(RESULTS_DIR)
REPS_PER_PERCENT = 50
KFOLD_REPS = 100
REPS_PER_PERCENT = 5
KFOLD_REPS = 10

PERCENTS = np.linspace(0, 1, 21)[1:-1]  # 5, 10, ..., 95
COLS = [f"{e:1.0f}" for e in PERCENTS * 100]
N_ROWS = len(COLS) * REPS_PER_PERCENT


def best_rect(m: int) -> Tuple[int, int]:
    """returns dimensions (smaller, larger) of closest rectangle"""
    low = int(np.floor(np.sqrt(m)))
    high = int(np.ceil(np.sqrt(m)))
    prods = [(low, low), (low, low + 1), (high, high), (high, high + 1)]
    for i, prod in enumerate(prods):
        if prod[0] * prod[1] >= m:
            return prod
    raise ValueError("Unreachable!")


def get_percent_acc_consistency(
    model: Type,
    model_args: Dict,
    x: ndarray,
    y: ndarray,
    percent: float,
    kfold_reps: int,
    x_test: ndarray = None,
    y_test: ndarray = None,
) -> Tuple[float, float]:
    x_down, _, y_down, _ = train_test_split(x, y, train_size=percent)
    if (x_test is not None) and (y_test is not None):
        errcon = ErrorConsistencyKFoldHoldout(
            model=model,
            x=x_down,
            y=y_down,
            n_splits=5,
            model_args=model_args,
            stratify=True,
            empty_unions="drop",
        )
        results = errcon.evaluate(
            x_test,
            y_test,
            repetitions=kfold_reps,
            save_test_accs=True,
            save_fold_accs=False,
            parallel_reps=True,
            loo_parallel=True,
            turbo=True,
            show_progress=False,
        )
        return np.mean(results.test_accs), np.mean(results.consistencies)
    errcon = ErrorConsistencyKFoldInternal(
        model=model,
        x=x_down,
        y=y_down,
        n_splits=5,
        model_args=model_args,
        stratify=True,
        empty_unions="drop",
    )
    results = errcon.evaluate(
        repetitions=kfold_reps,
        save_test_accs=True,
        save_fold_accs=False,
        parallel_reps=True,
        loo_parallel=True,
        turbo=True,
        show_progress=False,
    )
    return np.mean(results.test_accs), np.mean(results.consistencies)


# NOTE: for parkinsons and SPECT data, are going to get
# ValueError: n_splits=5 cannot be greater than the number of members in each class.
# for the small downsampling percents
def test_holdout_downsampling(capsys: Any) -> None:
    with capsys.disabled():
        for dataset_name, (x, y) in DATA.items():
            print(f"Preparing {dataset_name} data...")
            x, x_test, y, y_test = train_test_split(x, y, test_size=0.2)
            for classifier_name, (model, args) in CLASSIFIERS.items():
                print(f"Testing {classifier_name} classifier on {dataset_name} data...")
                data = np.full([N_ROWS, 3], -1, dtype=float)
                desc_percent = "Downsampling at {}%"
                pbar_percent = tqdm(desc=desc_percent.format(COLS[0]), total=len(COLS), leave=False)
                row = 0
                for percent, col in zip(PERCENTS, COLS):
                    pbar_percent.set_description(desc_percent.format(col))
                    desc_reps = "Repetition {:d}"
                    pbar_reps = tqdm(desc=desc_reps.format(0), total=REPS_PER_PERCENT, leave=False)
                    for r in range(REPS_PER_PERCENT):
                        pbar_reps.set_description(desc_reps.format(r))
                        acc, cons = get_percent_acc_consistency(
                            model, args, x, y, percent, KFOLD_REPS, x_test, y_test
                        )
                        data[row] = [percent, acc, cons]
                        row += 1
                        pbar_reps.update()
                    pbar_reps.close()
                    pbar_percent.update()
                pbar_percent.close()
                assert row == N_ROWS
                df = DataFrame(
                    data=data,
                    columns=["Percent", "Accuracy", "Consistency"],
                    index=range(N_ROWS),
                    dtype=float,
                )
                print(df)
                outfile = (
                    RESULTS_DIR
                    / f"{dataset_name}_{classifier_name}__k-fold-holdout_downsample.json"
                )
                df.to_json(outfile)


def plot_acc_cons(df: DataFrame, classifier: str, ax: plt.Axes) -> None:
    x = (df["Percent"] * 100).to_numpy().ravel()
    x_jitter = x + np.random.uniform(0, np.mean(np.diff(PERCENTS * 100)) / 2, len(x))
    y_acc, y_con = df["Accuracy"], df["Consistency"]
    acc_smooth = lowess(y_acc, x, return_sorted=False)
    con_smooth = lowess(y_con, x, return_sorted=False)
    args = dict(alpha=0.8, s=10, ax=ax, legend=None)
    sbn.scatterplot(x=x_jitter, y=y_acc, label="Accuracy", color="black", **args)
    sbn.scatterplot(x=x_jitter, y=y_con, label="Consistency", color="red", **args)
    sbn.lineplot(x=x, y=acc_smooth, color="black", lw=1, **args)
    sbn.lineplot(x=x, y=con_smooth, color="red", lw=1, **args)
    ax.set_xlabel("Downsampling percentage")
    ax.set_ylabel("Accuracy or Consistency")
    ax.set_title(f"{classifier} classifier")


def test_holdout_downsampling_plots(capsys: Any) -> None:
    files = sorted(RESULTS_DIR.rglob("Diabetes*.json"))
    datasets = [file.name[: file.name.find("_")] for file in files]
    classifiers = [file.name.split("_")[1] for file in files]
    dfs = [pd.read_json(file) for file in files]
    nrows, ncols = best_rect(len(dfs))
    sbn.set_style("darkgrid")
    fig: plt.Figure
    axes: plt.Axes
    ax: plt.Axes
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    count = 0
    for i, (df, classifier, ax) in enumerate(zip(dfs, classifiers, axes.flat)):
        plot_acc_cons(df, classifier, ax)
        count = i
    for j in range(count + 1, len(dfs) + 1):
        axes.flat[j].set_visible(False)
        fig.delaxes(axes.flat[j])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(
        f"Diabetes - Consistency and Accuracy across classifiers\n"
        f"{REPS_PER_PERCENT} repetitions per downsampling percentage\n"
        f"{KFOLD_REPS} k-fold repetitions for each consistency estimate"
    )
    fig.set_size_inches(h=2 * nrows + 3, w=4 * ncols + 1)
    # cleanup
    plt.show()
