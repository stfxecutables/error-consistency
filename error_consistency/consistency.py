from error_consistency.utils import to_numpy
import numpy as np

from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.cluster import KMeans

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from typing import cast, no_type_check
from typing_extensions import Literal

from error_consistency.model import Model, ModelFactory


PandasData = Union[DataFrame, Series]


class ErrorConsistency:
    """Compute error consistencies for a classifier.

    Parameters
    ----------
    model: Intersection[Callable, Type]
        A *class* where instances are classifiers that implement:

            1. a `.fit` or `.train` method that:
               - accepts predictors and targets, plus `fit_args`, and
               - updates the state of `model` when calling `.fit` or `.train`
            2. a `.predict` or `.test` method, that:
               - accepts testing samples, plus `predict_args`, and
               - requires having called `.fit` previously, and
               - returns *only* the predictions as a single ArrayLike
                 (e.g. NumPy array, List, pandas DataFrame or Series)

        E.g.

            import numpy as np
            from error_consistency import ErrorConsistency
            from sklearn.cluster import KNeighborsClassifier as KNN

            knn_args = dict(n_neighbors=5, n_jobs=1)
            errcon = ErrorConsistency(model=KNN, model_args=knn_args)

            # KNN is appropriate here because we could write e.g.
            x = np.random.uniform(0, 1, size=[100, 5])
            y = np.random.randint(0, 3, size=[100])
            x_test = np.random.uniform(0, 1, size=[20, 5])
            y_test = np.random.randint(0, 3, size=[20])

            KNN.fit(x, y)  # updates the state, no need to use a returned value
            y_pred = KNN.predict(x_test)  # returns a single object


    x: Union[List, pandas.DataFrame, pandas.Series, numpy.ndarray]
        ArrayLike object containing predictor samples. Must be in a format that is consumable with
        `model.fit(x, y, **model_args)` for arguments `model` and `model_args`. By default,
        splitting of x into cross-validation subsets will be along the first axis (axis 0), that is,
        the first axis is assumed to be the sample dimension. If your fit method requires a
        different sample dimension, you can specify this in `x_sample_dim`.

    y: Union[List, pandas.DataFrame, pandas.Series, numpy.ndarray]
        ArrayLike object containing targets. Must be in a format that is consumable with
        `model.fit(x, y, **model_args)` for arguments `model` and `model_args`. By default,
        splitting of y into cross-validation subsets will be along the first axis (axis 0), that is,
        the first axis is assumed to be the sample dimension. If your fit method requires a
        different sample dimension (e.g. y is a one-hot encoded array), you can specify this
        in `y_sample_dim`.

    model_args: Optional[Dict[str, Any]]
        Any arguments that are required each time to construct a fresh instance of the model (see
        above). Note that the data x and y must NOT be included here.

    fit_args: Optional[Dict[str, Any]]
        Any arguments that are required each time when calling the `.fit` or `.train` methods
        internally (see notes for `model` above). Note that the data x and y must NOT be included
        here.

    fit_args_x_y: Optional[Tuple[str, str]] = None
        Name of the arguments which data `x` and target `y` are passed to. This is needed because
        different libraries may have different conventions for how they expect predictors and
        targets to be passed in to `fit` or `train`.

        If None (default), it will be assumed that the `.fit` or `.train` method of the instance of
        `model` takes x as its first positional argument, and `y` as its second, as in e.g.
        `model.fit(x, y, **model_args)`.

        If a tuple of strings (x_name, y_name), then a dict will be constructed internally by
        splatting, e.g.

            args_dict = {**{x_name: x_train, y_name: y_train}, **model_args}
            model.fit(**args_dict)

    predict_args: Optional[Dict[str, Any]]
        Any arguments that are required each time when calling the `.predict` or `.test` methods
        internally (see notes for `model` above). Note that the data x must NOT be included here.

    predict_args_x: Optional[str] = None
        Name of the argument which data `x` is passed to during evaluation. This is needed because
        different libraries may have different conventions for how they expect predictors and
        targets to be passed in to `predict` or `test` calls.

        If None (default), it will be assumed that the `.predict` or `.test` method of the instance of
        `model` takes x as its first positional argument, as in e.g. `model.predict(x, **predict_args)`.

        If a string `x_name, then a dict will be constructed internally by splatting, e.g.

            args_dict = {**{x_name: x_train}, **model_args}
            model.predict(**args_dict)

    stratify: bool = False
        If True, use sklearn.model_selection.StratifiedKFold during internal k-fold. Otherwise, use
        sklearn.model_selection.KFold.

    x_sample_dim: int = 0
        The axis or dimension along which samples are indexed. Needed for splitting x into
        partitions for k-fold.

    y_sample_dim: int = 0
        The axis or dimension along which samples are indexed. Needed for splitting y into
        partitions for k-fold only if the target is e.g. one-hot encoded or dummy-coded.

    validate_model_argument: bool = True
        If True, create an unused instance of the model to check if it has appropriate methods
        (e.g. `.fit` or `.train` and `.predict` or `.test`)
    """

    def __init__(
        self,
        model: Any,
        x: ndarray,
        y: ndarray,
        x_test: ndarray = None,
        y_test: ndarray = None,
        model_args: Optional[Dict[str, Any]] = None,
        fit_args: Optional[Dict[str, Any]] = None,
        fit_args_x_y: Optional[Tuple[str, str]] = None,
        predict_args: Optional[Dict[str, Any]] = None,
        predict_args_x: Optional[str] = None,
        stratify: bool = False,
        x_sample_dim: int = 0,
        y_sample_dim: int = 0,
    ) -> None:
        self.model: Model
        self.stratify: bool
        self.x: ndarray
        self.y: ndarray
        self.x_df: Optional[PandasData]
        self.y_df: Optional[PandasData]
        self.x_test: ndarray
        self.y_test: ndarray
        self.x_test_df: Optional[PandasData]
        self.y_test_df: Optional[PandasData]

        self.model_factory = ModelFactory(
            model,
            model_args,
            fit_args,
            fit_args_x_y,
            predict_args,
            predict_args_x,
            x_sample_dim,
            y_sample_dim,
        )
        self.stratify = stratify

        self.x, self.y, self.x_df, self.y_df = self.__save_x_y(x, y)
        self.x_test, self.y_test, self.x_test_df, self.y_test_df = self.__save_x_y(x_test, y_test)

    @staticmethod
    def __save_x_y(x: ndarray, y: ndarray) -> Tuple[ndarray, ndarray, PandasData, PandasData]:
        x_df = x if isinstance(x, DataFrame) or isinstance(x, Series) else None
        y_df = y if isinstance(y, DataFrame) or isinstance(y, Series) else None
        x = to_numpy(x)
        y = to_numpy(y)
        return x, y, x_df, y_df
