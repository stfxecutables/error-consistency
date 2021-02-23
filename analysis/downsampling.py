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

sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis.constants import (
    PLOT_OUTDIR,
    OUTDIR,
    RESULTS_DIR,
    N_ROWS,
    REPS_PER_PERCENT,
    KFOLD_REPS,
    PERCENTS,
    COLS,
    DATASET_NAMES,
    KNN1_ARGS,
    KNN3_ARGS,
    KNN5_ARGS,
    KNN10_ARGS,
    LR_ARGS,
    SVC_ARGS,
    RF_ARGS,
    ADA_ARGS,
    MLP_ARGS,
    CLASSIFIERS,
    DATA,
)
from error_consistency.consistency import (
    ErrorConsistencyKFoldHoldout,
    ErrorConsistencyKFoldInternal,
)

CLASSIFIER_CHOICES = ["knn1", "knn3", "knn5", "knn10", "lr", "svm", "rf", "ada", "mlp"]
DATASET_CHOICES = ["diabetes", "park", "trans", "spect"]


def argparse_setup() -> ArgumentParser:
    parser = ArgumentParser(description="Plot effect of downsampling on error consistency.")
    parser.add_argument(
        "--plot-dir", type=Path, help="directory to save generated plots", default=PLOT_OUTDIR
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="directory to save generated data for plots",
        default=RESULTS_DIR,
    )
    parser.add_argument(
        "--kfold-reps", type=int, help="times to repeat k-fold per downsampling", default=KFOLD_REPS
    )
    parser.add_argument(
        "--percent-reps",
        type=int,
        help="times to downsample for each percentage",
        default=REPS_PER_PERCENT,
    )
    parser.add_argument(
        "--percent-spacing",
        type=int,
        help="space between percents. E.g.  5 means 5, 10, 15, ...",
        default=5,
    )
    parser.add_argument("--classifier", choices=CLASSIFIER_CHOICES, required=True)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--pbar", action="store_true")
    parser.add_argument(
        "--cpus", type=int, help="number of cpus to use for parallelization", default=4
    )
    return parser


def classifier_from_args(args: Namespace) -> str:
    mappings = dict(
        knn1="KNN-1",
        knn3="KNN-3",
        knn5="KNN-5",
        knn10="KNN-10",
        lr="Logistic Regression",
        svm="SVM Classifier",
        rf="Random Forest",
        ada="AdaBoosted DTree",
        mlp="MLP",
    )
    return mappings[args.classifier]


def dataset_from_args(args: Namespace) -> str:
    mappings = dict(diabetes="Diabetes", park="Parkinsons", trans="Transfusion", spect="SPECT")
    return mappings[args.dataset]


def get_percent_acc_consistency(
    model: Type,
    model_args: Dict,
    x: ndarray,
    y: ndarray,
    percent: float,
    kfold_reps: int,
    x_test: ndarray = None,
    y_test: ndarray = None,
    cpus: int = 4,
) -> Tuple[float, float]:
    x_down, y_down = None, None
    if percent >= 100:
        x_down, y_down = x, y
    else:
        for _ in range(100):
            try:
                x_down, _, y_down, _ = train_test_split(x, y, stratify=y, train_size=percent / 100)
                break
            except Exception:
                pass
    if x_down is None:
        raise RuntimeError("Failed to downsample")
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
            parallel_reps=cpus,
            loo_parallel=cpus,
            turbo=True,
            show_progress=False,
        )
        return np.mean(results.test_accs), np.mean(results.consistencies)
    errcon_internal = ErrorConsistencyKFoldInternal(
        model=model,
        x=x_down,
        y=y_down,
        n_splits=5,
        model_args=model_args,
        stratify=True,
        empty_unions="drop",
    )
    results = errcon_internal.evaluate(
        repetitions=kfold_reps,
        save_test_accs=True,
        save_fold_accs=False,
        parallel_reps=cpus,
        loo_parallel=cpus,
        turbo=True,
        show_progress=False,
    )
    return np.mean(results.test_accs), np.mean(results.consistencies)


# NOTE: for parkinsons and SPECT data, are going to get
# ValueError: n_splits=5 cannot be greater than the number of members in each class.
# for the small downsampling percents
def holdout_downsampling_all(show_progress: bool = False) -> None:
    disable = not show_progress
    for dataset_name, (x, y) in DATA.items():
        print(f"Preparing {dataset_name} data...")
        x, x_test, y, y_test = train_test_split(x, y, test_size=0.2)
        for classifier_name, (model, args) in CLASSIFIERS.items():
            print(f"Testing {classifier_name} classifier on {dataset_name} data...")
            data = np.full([N_ROWS, 3], -1, dtype=float)
            desc_percent = "Downsampling at {}%"
            pbar_percent = tqdm(
                desc=desc_percent.format(COLS[0]), total=len(COLS), leave=False, disable=disable
            )
            row = 0
            for percent, col in zip(PERCENTS, COLS):
                pbar_percent.set_description(desc_percent.format(col))
                desc_reps = "Repetition {:d}"
                pbar_reps = tqdm(
                    desc=desc_reps.format(0), total=REPS_PER_PERCENT, leave=False, disable=disable
                )
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
                RESULTS_DIR / f"{dataset_name}_{classifier_name}__k-fold-holdout_downsample.json"
            )
            df.to_json(outfile)


def holdout_downsampling(args: Namespace,) -> None:
    disable = not args.pbar
    dataset = dataset_from_args(args)
    classifier = classifier_from_args(args)
    d = float(args.percent_spacing)
    percents = np.arange(0, 100 + d, d)[1:]
    percents = percents[percents >= 5]
    cols = [f"{e:1.0f}" for e in percents]
    reps_per_percent = args.percent_reps
    kfold_reps = args.kfold_reps
    n_rows = len(percents) * reps_per_percent
    outdir = args.results_dir
    if not outdir.exists():
        os.makedirs(outdir)
    cpus = args.cpus

    x, y = DATA[dataset]
    print(f"Preparing {dataset} data...")
    x, x_test, y, y_test = train_test_split(x, y, test_size=0.2)
    model, model_args_dict = CLASSIFIERS[classifier]
    model_args = model_args_dict[dataset]
    print(f"Testing {classifier} classifier on {dataset} data...")
    data = np.full([n_rows, 3], -1, dtype=float)
    desc_percent = "Downsampling at {}%"
    pbar_percent = tqdm(
        desc=desc_percent.format(cols[0]), total=len(cols), leave=False, disable=disable
    )
    row = 0
    for percent, col in zip(percents, cols):
        pbar_percent.set_description(desc_percent.format(col))
        desc_reps = "Repetition {:d}"
        pbar_reps = tqdm(
            desc=desc_reps.format(0), total=reps_per_percent, leave=False, disable=disable
        )
        for r in range(reps_per_percent):
            pbar_reps.set_description(desc_reps.format(r))
            acc, cons = get_percent_acc_consistency(
                model, model_args, x, y, percent, kfold_reps, x_test, y_test, cpus
            )
            data[row] = [percent, acc, cons]
            row += 1
            pbar_reps.update()
        pbar_reps.close()
        pbar_percent.update()
    pbar_percent.close()
    print("row:", row)
    assert row == n_rows
    df = DataFrame(
        data=data, columns=["Percent", "Accuracy", "Consistency"], index=range(n_rows), dtype=float
    )
    print(df)
    classifier = classifier.replace(" ", "_")
    outfile = outdir / f"{dataset}_{classifier}__k-fold-holdout_downsample.json"
    df.to_json(outfile)


def generate_arguments(
    results_dir: Path = RESULTS_DIR,
    kfold_reps: int = KFOLD_REPS,
    percent_reps: int = REPS_PER_PERCENT,
    cpus: int = 4,
) -> str:
    template = "--classifier={classifier} --dataset={dataset} --kfold-reps={kfold_reps} --percent-reps={percent_reps} --results-dir={results_dir} --cpus={cpus}"
    lines = []
    for dataset in DATASET_CHOICES:
        for classifier in CLASSIFIER_CHOICES:
            lines.append(
                template.format(
                    classifier=classifier,
                    dataset=dataset,
                    kfold_reps=kfold_reps,
                    percent_reps=percent_reps,
                    results_dir=results_dir,
                    cpus=cpus,
                )
            )
    return "\n".join(lines)


if __name__ == "__main__":
    # print(generate_arguments(percent_reps=50, kfold_reps=100))
    parser = argparse_setup()
    args = parser.parse_args()
    # args = parser.parse_args(
    #     "--classifier lr --dataset diabetes --kfold-reps 100 --percent-reps 50 --results-dir analysis/results/testresults --pbar --cpus 8".split(
    #         " "
    #     )
    # )
    filterwarnings("ignore", message="Got `batch_size`", category=UserWarning)
    filterwarnings("ignore", message="Stochastic Optimizer")
    filterwarnings("ignore", message="Liblinear failed to converge")
    holdout_downsampling(args)

