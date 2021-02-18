import sys
import pytest
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from numpy import ndarray
from pandas import DataFrame, Series
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing import cast, no_type_check
from typing_extensions import Literal

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.model_selection import train_test_split

from error_consistency.consistency import (
    ErrorConsistencyKFoldHoldout,
    ErrorConsistencyKFoldInternal,
)
from tests.loading import load_diabetes, load_park, load_trans, load_SPECT


KNN1_ARGS = dict(n_neighbors=1)
KNN3_ARGS = dict(n_neighbors=3)
KNN5_ARGS = dict(n_neighbors=5)
KNN10_ARGS = dict(n_neighbors=10)
LR_ARGS = dict(solver="liblinear")  # small datasets
SVC_ARGS = dict()
RF_ARGS = dict()
ADA_ARGS = dict()

CLASSIFIERS: Dict[str, Tuple[Type, Dict]] = {
    "KNN-1": (KNN, KNN1_ARGS),
    "KNN-3": (KNN, KNN3_ARGS),
    "KNN-5": (KNN, KNN5_ARGS),
    "KNN-10": (KNN, KNN10_ARGS),
    "Logistic Regression": (LR, LR_ARGS),
    "SVM": (SVC, SVC_ARGS),
    "Random Forest": (RF, RF_ARGS),
    "AdaBoosted DTree": (AdaBoost, ADA_ARGS),
}
DATA: Dict[str, Tuple[DataFrame, DataFrame]] = {
    "Diabetes": load_diabetes(),
    "Parkinsons": load_park(),
    "Transfusion": load_trans(),
    "SPECT": load_SPECT(),
}
OUTDIR = Path(__file__).resolve().parent / "results"


def test_classifiers_holdout(capsys: Any) -> None:
    with capsys.disabled():
        for dataset_name, (x, y) in DATA.items():
            print(f"Preparing {dataset_name} data...")
            x, x_test, y, y_test = train_test_split(x, y, test_size=0.2)
            df_data = pd.DataFrame()
            for classifier_name, (model, args) in CLASSIFIERS.items():
                print(f"Testing {classifier_name} classifier on {dataset_name} data...")
                errcon = ErrorConsistencyKFoldHoldout(
                    model=model,
                    x=x,
                    y=y,
                    n_splits=5,
                    model_args=args,
                    stratify=True,
                    empty_unions="drop",
                )
                results = errcon.evaluate(
                    x_test,
                    y_test,
                    repetitions=500,
                    save_test_accs=True,
                    save_fold_accs=True,
                    parallel_reps=True,
                    loo_parallel=True,
                    turbo=True,
                )
                print(results)
                df_classifier = results.summary(classifier_name)
                df_data = df_data.append(df_classifier)
                print(df_data)
            print(df_data)
            outfile = OUTDIR / f"{dataset_name}__k-fold-holdout.csv"
            df_data.to_csv(outfile)


def test_classifiers_internal(capsys: Any) -> None:
    with capsys.disabled():
        for dataset_name, (x, y) in DATA.items():
            print(f"Preparing {dataset_name} data...")
            df_data = pd.DataFrame()
            for classifier_name, (model, args) in CLASSIFIERS.items():
                print(f"Testing {classifier_name} classifier on {dataset_name} data...")
                errcon = ErrorConsistencyKFoldInternal(
                    model=model,
                    x=x,
                    y=y,
                    n_splits=5,
                    model_args=args,
                    stratify=True,
                    empty_unions="drop",
                )
                results = errcon.evaluate(
                    repetitions=500,
                    save_test_accs=True,
                    save_fold_accs=True,
                    parallel_reps=True,
                    loo_parallel=True,
                    turbo=True,
                )
                print(results)
                df_classifier = results.summary(classifier_name)
                df_data = df_data.append(df_classifier)
                print(df_data)
            print(df_data)
            outfile = OUTDIR / f"{dataset_name}__k-fold-internal.csv"
            df_data.to_csv(outfile)
