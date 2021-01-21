import numpy as np
import random
import sys

from dataclasses import dataclass
from numpy import ndarray
from numpy.random import SeedSequence, MT19937, RandomState
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.cluster import KMeans

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from typing import cast, no_type_check
from typing_extensions import Literal

from error_consistency.model import Model, ModelFactory
from error_consistency.utils import to_numpy


PandasData = Union[DataFrame, Series]
Shape = Tuple[int, ...]


@dataclass(eq=False)
class ConsistencyResults:
    """Holds results from evaluating error consistency.

    Properties
    ----------
    consistencies: ndarray
        A one-dimensional numpy array all calculated error consistencies.

    consistency_matrices: List[ndarray]
        Given n repetitions of the k-fold process, a list of n square matrices of size k x k, with
        error consistencies between folds on the upper diagonal, and zeroes else. That is, if we
        denote one of these matrices as A, then `A[i,j] == 0` if i >= j, and `A[i, j]` is equal to
        the error consistency between the folds i and j, for i, j in {0, 1, ..., k - 1} .

    scores: Optional[ndarray] = None
        A one-dimensional numpy array of all calculated accuracies.

    error_arrays: Optional[List[ndarray]] = None
        A list of the boolean error arrays (`y_pred_i == y_test` for fold `i`) for all repetitions.

    predictions: Optional[List[ndarray]] = None
        A list of the actual predicted values across all reptitions and folds.

    models: Optional[List[Ant]] = None
        A list of all fitted models across repetitions and folds.
    """

    consistencies: ndarray
    consistency_matrices: List[ndarray]
    scores: Optional[ndarray] = None
    error_arrays: Optional[List[ndarray]] = None
    predictions: Optional[List[ndarray]] = None
    models: Optional[List[Any]] = None


class ErrorConsistencyKFold:
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

    n_splits: int = 5
        How many folds to use for validating error consistency.

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

    onehot_y: bool = True
        Only relevant for two-dimensional `y`. Set to True if `y` is a one-hot array with samples
        indexed by `y_sample_dim`. Set to False if `y` is dummy-coded.

    """

    def __init__(
        self,
        model: Any,
        x: ndarray,
        y: ndarray,
        n_splits: int = 5,
        model_args: Optional[Dict[str, Any]] = None,
        fit_args: Optional[Dict[str, Any]] = None,
        fit_args_x_y: Optional[Tuple[str, str]] = None,
        predict_args: Optional[Dict[str, Any]] = None,
        predict_args_x: Optional[str] = None,
        stratify: bool = False,
        x_sample_dim: int = 0,
        y_sample_dim: int = 0,
        onehot_y: bool = True,
    ) -> None:
        self.model: Model
        self.stratify: bool
        self.n_splits: int
        self.x: ndarray
        self.y: ndarray
        self.x_sample_dim: int
        self.y_sample_dim: int
        # if user is using DataFrames, save reference to these for the variable names
        self.x_df: Optional[PandasData]
        self.y_df: Optional[PandasData]

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
        if n_splits < 2:
            raise ValueError("Must have more than one split for K-fold.")
        self.n_splits = n_splits
        if not onehot_y:
            raise NotImplementedError("Dummy-coded `y` currently not implemented.")
        self.onehot_y = onehot_y

        self.x, self.y, self.x_df, self.y_df = self.__save_x_y(x, y)
        self.x_transpose_shape, self.y_transpose_shape, self.x_sample_dim, self.y_sample_dim = self.__save_dims(
            x, y, x_sample_dim, y_sample_dim
        )

    def evaluate(
        self,
        x_test: ndarray = None,
        y_test: ndarray = None,
        repetitions: int = 5,
        compute_scores: bool = True,
        save_error_arrays: bool = False,
        save_predictions: bool = False,
        save_models: bool = False,
        seed: int = None,
    ) -> ConsistencyResults:
        """Evaluate the error consistency of the classifier.

        Parameters
        ----------
        x_test: Optional[Union[List, pandas.DataFrame, pandas.Series, numpy.ndarray]] = None
            ArrayLike object containing holdout predictor samples that the model will never be
            trained or fitted on. Must be have a format identical to that of `x` passed into
            constructor (see above).

        y_test: Optional[Union[List, pandas.DataFrame, pandas.Series, numpy.ndarray]] = None
            ArrayLike object containing holdout target values that the model will never be trained
            or fitted on. Must be have a format identical to that of `x` passed into constructor
            (see above).

        repetitions: int = 5
            How many times to repeat the k-fold process. Yields `k*repetitions` error consistencies
            if both `x_test` and `y_test` are proided, and `repetitions*(repititions - 1)/2`
            consistencies otherwise. Note that if both `x_test` and `y_test` are not provided, then
            setting repetitions to 1 will raise an error, since this results in insufficient arrays
            to compare errors.

        compute_scores: bool = True
            If True (default) also compute accuracy scores for each fold and save them in
            `results.scores`. If False, `x_test` and `y_test` are not None, only compute predictions
            on `x_test`, and not testing subsets of each k-fold partition. Useful if prediction is
            expensive or otherwise not needed.

        save_error_arrays: bool = False,
            If True also save the boolean arrays indicating the error locations of each fold in
            `results.error_arrays`. If False (default), leave this property empty in the results.

        save_predictions: bool = False
            If True also save the predicted target values of each fold in `results.predictions`.
            If False (default), leave this property empty in the results.

        save_models: bool = False,
            If True also save the fitted models of each fold in `results.models`.
            If False (default), leave this property empty in the results.

        seed: int = None,

        Returns
        -------
        results: ConsistencyResults
            A dataclass with properties:

                consistencies: ndarray
                consistency_matrices: List[ndarray]
                scores: Optional[ndarray] = None
                error_arrays: Optional[List[ndarray]] = None
                predictions: Optional[List[ndarray]] = None
                models: Optional[List[Any]] = None
        """

        def array_indexer(array: ndarray, idx: ndarray) -> ndarray:
            # grotesque but hey, we need to index into a position programmatically
            colons = [":" for _ in range(self.x.ndim)]
            colons[self.x_sample_dim] = "idx"
            idx_string = f"{','.join(colons)}"
            return eval(f"array[{idx_string}]")

        self.x_test, self.y_test = self.__save_x_y(x_test, y_test)[0:2]
        if seed is not None:
            random.seed(seed)
            rng = np.random.default_rng(seed)
            seeds = rng.integers(0, sys.maxsize, repetitions)
        else:
            rng = np.random.default_rng()
            seeds = rng.integers(0, sys.maxsize, repetitions)

        kfolds = [KFold(n_splits=self.n_splits, shuffle=True, random_state=seed) for seed in seeds]

        consistencies: List[float] = []
        consistency_matrices: List[ndarray] = []
        scores: Optional[ndarray] = [] if compute_scores else None
        error_arrays: Optional[List[ndarray]] = [] if save_error_arrays else None
        predictions: Optional[List[ndarray]] = [] if save_predictions else None
        models: Optional[List[Any]] = [] if save_models else None
        idx = np.arange(0, int(self.x.shape[self.x_sample_dim]), dtype=int)
        for kfold in kfolds:
            train_idx, test_idx = kfold.split(idx)
            x_train_fold = array_indexer(self.x, train_idx)
            y_train_fold = array_indexer(self.y, train_idx)

            x_test_fold = array_indexer(self.x, test_idx)
            y_test_fold = array_indexer(self.y, test_idx)
            model = self.model_factory.create()
            model.fit(x_train_fold, y_train_fold)
            model.predict()

        raise NotImplementedError()

    @staticmethod
    def __save_x_y(x: ndarray, y: ndarray) -> Tuple[ndarray, ndarray, PandasData, PandasData]:
        x_df = x if isinstance(x, DataFrame) or isinstance(x, Series) else None
        y_df = y if isinstance(y, DataFrame) or isinstance(y, Series) else None
        x = to_numpy(x)
        y = to_numpy(y)
        if y.ndim > 2:
            raise ValueError("Target `y` can only be 1-dimensional or two-dimensional.")
        if y.ndim == 2:
            uniques = np.unique(y.ravel()).astype(int)
            if not np.array_equal(uniques, [0, 1]):
                raise ValueError
        return x, y, x_df, y_df

    @staticmethod
    def __save_dims(
        x: ndarray, y: ndarray, x_sample_dim: int, y_sample_dim: int
    ) -> Tuple[Shape, Shape, int, int]:
        # we need to convert the sample dimensions to positive values to construct transpose shapes
        if y_sample_dim > 1 or y_sample_dim < -1:
            raise ValueError(
                "Invalid `y_sample_dim`. Must be 0 for one-dimensional `y`, "
                "and either 1 or -1 for two-dimensional `y`."
            )
        y_dim = int(np.abs(y_sample_dim))  # 1, -1 have same behaviour if dim=2, abs(0) is 0

        if (x_sample_dim > x.ndim - 1) or (x_sample_dim < -x.ndim):
            raise ValueError(
                "Invalid `x_sample_dim`. `x_sample_dim` must satisfy "
                "`x.ndim - 1 < x_sample_dim < -x.ndim`"
            )
        x_dim = x.ndim - x_sample_dim if x_sample_dim < 0 else x_sample_dim

        xdim_indices = list(range(int(x.ndim)))
        x_transpose_shape = (x_dim, *xdim_indices[:x_dim], *xdim_indices[x_dim + 1 :])
        y_transpose_shape = y.shape if y.ndim != 2 else ((0, 1) if y_dim == 1 else (1, 0))
        return x_transpose_shape, y_transpose_shape, x_dim, y_dim


class ErrorConsistencyMonteCarlo:
    """Calculate error consistency using repeated random train/test splits."""
