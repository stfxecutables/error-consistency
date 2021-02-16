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


# fmt: off
DATAROOT = Path(__file__).resolve().parent / "datasets"

DIABETES    = DATAROOT / "diabetes/diabetes.csv"  # noqa
PARK        = DATAROOT / "parkinsons/parkinsons.data"  # noqa
SPECT_TRAIN = DATAROOT / "SPECT/SPECT.train"  # noqa
SPECT_TEST  = DATAROOT / "SPECT/SPECT.test"  # noqa
TRANSFUSION = DATAROOT / "transfusion/transfusion.data"  # noqa

DIABETES_TARGET = "Outcome"  # noqa
PARK_TARGET     = "status"  # noqa
SPECT_TARGET    = "diagnosis"  # noqa
TRANS_TARGET    = "donated"  # original dataset has an entire sentence for the column name... # noqa
# fmt: on


def load_df(path: Path, target: str) -> Tuple[DataFrame, Series]:
    df = pd.read_csv(path)
    x = df.drop(columns=[target])
    y = df[target].copy()
    return x, y


def load_diabetes() -> Tuple[DataFrame, Series]:
    return load_df(DIABETES, DIABETES_TARGET)


def load_park() -> Tuple[DataFrame, Series]:
    return load_df(PARK, PARK_TARGET)


def load_trans() -> Tuple[DataFrame, Series]:
    return load_df(TRANSFUSION, TRANS_TARGET)


def load_SPECT() -> Tuple[DataFrame, Series]:
    x1, y1 = load_df(SPECT_TRAIN, SPECT_TARGET)
    x2, y2 = load_df(SPECT_TEST, SPECT_TARGET)
    return pd.concat([x1, x2], axis=0), pd.concat([y1, y2], axis=0)
