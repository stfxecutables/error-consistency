import numpy as np

from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.cluster import KMeans

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from typing import cast, no_type_check
from typing_extensions import Literal


PandasData = Union[DataFrame, Series]


class ErrorConsistency:
    """Compute error consistencies.

    Parameters
    ----------
    model: Intersection[Callable, Type]
        A *class* for which instances implement (1) .fit or .train methods and (2) .predict or .test
        method, and which takes `model_args` in its constructor. E.g.

            from error_consistency import ErrorConsistency
            from sklearn.cluster import KNeighborsClassifier as KNN

            knn_args = dict(n_neighbors=5, n_jobs=1)
            errcon = ErrorConsistency(model=KNN, model_args=knn_args)

    X: Union[List, pandas.DataFrame, pandas.Series, numpy.ndarray]
        ArrayLike object containing predictor samples. Must be in a format that is consumable with
        `model.fit(X, y, **model_args)` for arguments `model` and `model_args`. By default,
        splitting of X into cross-validation subsets will be along the first axis (axis 0), that is,
        the first axis is assumed to be the sample dimension. If your fit method requires a
        different sample dimension, you can specify this in `X_sample_dim`.

    y: Union[List, pandas.DataFrame, pandas.Series, numpy.ndarray]
        ArrayLike object containing targets


    model_args: Optional[Dict[str, Any]]
        Any arguments that are required each time to construct a fresh instance of the model (see
        above). Note that the data X and y must NOT be included here.

    fit_args: Optional[Dict[str, Any]]
        Any arguments that may



    Returns
    -------
    val1: Any
    """

    def __init__(
        self,
        model: Any,
        X: ndarray,
        y: ndarray,
        model_args: Dict[str, Any] = None,
        fit_args: Optional[Dict[str, Any]] = None,
        fit_args_x: str = "X",
        fit_args_y: str = "y",
        predict_args: Optional[Dict[str, Any]] = None,
        predict_args_x: str = "x",
        stratify: bool = False,
        X_sample_dim: int = 0,
        validate_model_argument: bool = True,
    ) -> None:
        self.model: Any
        self.model_args: Optional[Dict[str, Any]]
        self.fit_args: Optional[Dict[str, Any]]
        self.fit_args_x: str
        self.fit_args_y: str
        self.predict_args: Optional[Dict[str, Any]]
        self.predict_args_x: str
        self.stratify: bool
        self.x: ndarray
        self.y: ndarray
        self.x_df: Optional[PandasData]
        self.y_df: Optional[PandasData]

        self.model = self.__validate_model(model, model_args) if validate_model_argument else model
        self.model_args = self.__validate_args(model_args, "model_args")
        self.fit_args = self.__validate_args(fit_args, "fit_args")
        self.fit_args_x = fit_args_x
        self.fit_args_y = fit_args_y
        self.predict_args = self.__validate_args(predict_args, "predict_args")
        self.predict_args_x = predict_args_x
        self.stratify = stratify

        self.x, self.y, self.x_df, self.y_df = self.__save_x_y(X, y)

    @staticmethod
    def __validate_model(model: Any, model_args: Dict[str, Any]) -> Any:
        if not (isinstance(model, type) and callable(model)):
            raise ValueError(
                "Argument `model` must be a class that constructs a new model instance when called"
                " with `model_args`. Example:"
                "\n"
                "    from error_consistency import ErrorConsistency\n"
                "    from sklearn.cluster import KNeighborsClassifier as KNN\n\n"
                "    knn_args = dict(n_neighbors=5, n_jobs=1)"
                "    errcon = ErrorConsistency(model=KNN, model_args=knn_args)\n"
            )
        try:
            instance = model(**model_args)
        except Exception as e:
            raise ValueError(f"Cannot create model: {model} with model_args: {model_args}") from e
        if not (hasattr(instance, "fit") or hasattr(instance, "train")):
            raise ValueError(
                "Argument `model` must be class for which instances implement "
                "either a `fit` or `train` method."
            )
        if not (hasattr(instance, "predict") or hasattr(instance, "test")):
            raise ValueError(
                "Argument `model` must be class for which instances implement "
                "either a `predict` or `test` method."
            )
        return model

    @staticmethod
    def __save_xy(X: ndarray, y: ndarray) -> Tuple[ndarray, ndarray, PandasData, PandasData]:
        if isinstance(X, DataFrame) or isinstance(X, Series):
            x_df = X
            x: ndarray = X.to_numpy()
        elif isinstance(X, ndarray):
            x_df = None
            x = X
        elif isinstance(X, list):
            x_df = None
            x = np.array(X)
        else:
            raise ValueError("Unsupported type for predictor `X`")

        if x.ndim > 2:
            raise


        if isinstance(y, DataFrame) or isinstance(y, Series):
            y_df = y
            y: ndarray = y.to_numpy()
        elif isinstance(y, ndarray):
            y_df = None
        elif isinstance(y, list):
            y_df = None
            y = np.array(y)
        else:
            raise ValueError("Unsupported type for target `y`")

        return x, y, x_df, y_df

    @staticmethod
    def __validate_args(args: Optional[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
        """Ensure args are a Dict"""
        if args is None:
            return args
        if not isinstance(args, dict):
            raise ValueError(f"Argument `{name}` is not a Python dict.")
        for key in args.keys():
            if not isinstance(key, str):
                raise ValueError(f"Argument `{name}` must be a dict where all keys are strings.")
            if key.lower() == "x" or key.lower() == "y":
                for data_type, typename in [
                    (int, "int"),
                    (float, "float"),
                    (ndarray, "numpy.ndarray"),
                    (DataFrame, "pandas.DataFrame"),
                    (Series, "pandas.Series"),
                ]:
                    if isinstance(args[key], data_type):
                        raise ValueError(
                            f"Found argument in `{name}` with name 'x', 'X', 'y', or 'Y' and of "
                            f"type {typename}. Calculating error consistency requires "
                            "instantiating a new model and training and testing on different "
                            "subsets of the predictor X and target y, thus the same data can not "
                            "be passed into e.g. `.fit` or `.predict` each time. You should thus "
                            "specify only the *name* of the arguments that will be passed the X "
                            "and y values in e.g. `fit_args_x`, `fit_args_y`, and `predict_args_x`."
                        )
        return args
