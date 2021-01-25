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


class Model:
    """Helper class for making a unified interface to different model types.

    Parameters
    ----------
    model: Intersection[Callable, Type]
        A *class* for which instances implement (1) .fit or .train methods and (2) .predict or .test
        method, and which takes `model_args` in its constructor. E.g.

            from error_consistency import ErrorConsistency
            from sklearn.cluster import KNeighborsClassifier as KNN

            knn_args = dict(n_neighbors=5, n_jobs=1)
            errcon = ErrorConsistency(model=KNN, model_args=knn_args)

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

        If None (default), it will be assumed that the `.predict` or `.test` method of the instance
        of `model` takes x as its first positional argument, as in e.g.
        `model.predict(x, **predict_args)`.

        If a string `x_name, then a dict will be constructed internally by splatting, e.g.

            args_dict = {**{x_name: x_train}, **model_args}
            model.predict(**args_dict)

    x_sample_dim: int = 0
        The axis or dimension along which samples are indexed. Needed for splitting x into
        partitions for k-fold.

    y_sample_dim: int = 0
        The axis or dimension along which samples are indexed. Needed for splitting y into
        partitions for k-fold only if the target is e.g. one-hot encoded or dummy-coded.
    """

    def __init__(
        self,
        model: Any,
        model_args: Optional[Dict[str, Any]] = None,
        fit_args: Optional[Dict[str, Any]] = None,
        fit_args_x_y: Optional[Tuple[str, str]] = None,
        predict_args: Optional[Dict[str, Any]] = None,
        predict_args_x: Optional[str] = None,
        s_sample_dim: int = 0,
        y_sample_dim: int = 0,
    ) -> None:
        # We don't validate args, that is handled by the Model Factory
        self.model = model
        self.model_args = model_args
        self.fit_args = fit_args
        self.fit_args_x_y = fit_args_x_y
        self.predict_args = predict_args
        self.predict_args_x = predict_args_x
        self.x_sample_dim = s_sample_dim
        self.y_sample_dim = y_sample_dim

    def fit(self, x: ndarray, y: ndarray) -> None:
        fit_args = {} if self.fit_args is None else self.fit_args

        if hasattr(self.model, "fit"):
            if self.fit_args_x_y is None:
                self.model.fit(x, y, **fit_args)
            else:
                x_y_args = {self.fit_args_x_y[0]: x, self.fit_args_x_y[1]: y}
                self.model.fit(**x_y_args, **fit_args)
        elif hasattr(self.model, "train"):
            if self.fit_args_x_y is None:
                self.model.train(x, y, **fit_args)
            else:
                x_y_args = {self.fit_args_x_y[0]: x, self.fit_args_x_y[1]: y}
                self.model.train(**x_y_args, **fit_args)
        else:
            raise ValueError(
                "Bug in model validation code (no `fit` or `train` method found.) Please report this."
            )

    def predict(self, x: ndarray) -> ndarray:
        predict_args = {} if self.predict_args is None else self.predict_args

        if hasattr(self.model, "predict"):
            if self.predict_args_x is None:
                results = self.model.predict(x, **predict_args)
            else:
                x_args = {self.predict_args_x[0]: x}
                results = self.model.predict(**x_args, **predict_args)
        elif hasattr(self.model, "test"):
            if self.predict_args_x is None:
                results = self.model.test(x, **predict_args)
            else:
                x_args = {self.predict_args_x[0]: x}
                results = self.model.test(**x_args, **predict_args)
        else:
            raise ValueError(
                "Bug in model validation code (no `predict` or `test` method found.) Please report this."
            )
        try:
            return to_numpy(results)
        except ValueError as e:
            raise RuntimeError(
                "Method `.predict` returns an object or objects that cannot be converted to NumPy."
            ) from e


class ModelFactory:
    """Helper class for making a unified interface to different model types.

    Parameters
    ----------
    model: Intersection[Callable, Type]
        A *class* for which instances implement (1) .fit or .train methods and (2) .predict or .test
        method, and which takes `model_args` in its constructor. E.g.

            from error_consistency import ErrorConsistency
            from sklearn.cluster import KNeighborsClassifier as KNN

            knn_args = dict(n_neighbors=5, n_jobs=1)
            errcon = ErrorConsistency(model=KNN, model_args=knn_args)

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

        If None (default), it will be assumed that the `.predict` or `.test` method of the instance
        of `model` takes x as its first positional argument, as in e.g.
        `model.predict(x, **predict_args)`.

        If a string `x_name, then a dict will be constructed internally by splatting, e.g.

            args_dict = {**{x_name: x_train}, **model_args}
            model.predict(**args_dict)

    x_sample_dim: int = 0
        The axis or dimension along which samples are indexed. Needed for splitting x into
        partitions for k-fold.

    y_sample_dim: int = 0
        The axis or dimension along which samples are indexed. Needed for splitting y into
        partitions for k-fold only if the target is e.g. one-hot encoded or dummy-coded.
    """

    def __init__(
        self,
        model: Any,
        model_args: Optional[Dict[str, Any]] = None,
        fit_args: Optional[Dict[str, Any]] = None,
        fit_args_x_y: Optional[Tuple[str, str]] = None,
        predict_args: Optional[Dict[str, Any]] = None,
        predict_args_x: Optional[str] = None,
        s_sample_dim: int = 0,
        y_sample_dim: int = 0,
    ) -> None:
        self.constructor = self.__validate_model(model)
        self.model_args = self.__validate_args(model_args, "model_args")
        self.fit_args = self.__validate_args(fit_args, "fit_args")
        self.fit_args_x_y = fit_args_x_y
        self.predict_args = self.__validate_args(predict_args, "predict_args")
        self.predict_args_x = predict_args_x
        self.x_sample_dim = s_sample_dim
        self.y_sample_dim = y_sample_dim

    def create(self) -> Model:
        try:
            model = self.constructor(**self.model_args)
        except Exception as e:
            raise ValueError(
                f"Cannot create model: {self.constructor} with model_args: {self.model_args}"
            ) from e
        return Model(
            model,
            self.model_args,
            self.fit_args,
            self.fit_args_x_y,
            self.predict_args,
            self.predict_args_x,
            self.x_sample_dim,
            self.y_sample_dim,
        )

    @staticmethod
    def __validate_model(model: Any) -> Any:
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
        if not (hasattr(model, "fit") or hasattr(model, "train")):
            raise ValueError(
                "Argument `model` must be class for which instances implement "
                "either a `fit` or `train` method."
            )
        # for methodname in ["fit", "train", "predict", "test"]:
        #     if not callable(getattr(model, methodname, None)):
        #         raise ValueError(
        #             f"`model` has a `.{methodname}` property but `.{methodname}` is not callable."
        #         )
        return model

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
                            "subsets of the predictor x and target y, thus the same data can not "
                            "be passed into e.g. `.fit` or `.predict` each time. You should thus "
                            "specify only the *name* of the arguments that will be passed the x "
                            "and y values in e.g. `fit_args_x`, `fit_args_y`, and `predict_args_x`."
                        )
        return args
