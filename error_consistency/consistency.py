from concurrent.futures import process
import numpy as np
import random
import sys

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy import ndarray
from numpy.random import SeedSequence, MT19937, RandomState
from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.cluster import KMeans
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool, cpu_count


from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from typing import cast, no_type_check
from typing_extensions import Literal

from error_consistency.functional import UnionHandling, get_y_error, error_consistencies
from error_consistency.model import Model, ModelFactory
from error_consistency.utils import to_numpy


PandasData = Union[DataFrame, Series]
Shape = Tuple[int, ...]


@dataclass(eq=False)
class KFoldResults:
    fitted_model: Model
    score: Optional[ndarray]
    prediction: ndarray


@dataclass(eq=False)
class ConsistencyResults:
    """Holds results from evaluating error consistency.

    Properties
    ----------
    consistencies: ndarray
        Given n repetitions of the k-fold process, an array C of shape (n, k, k), an array of square
        matrices of size k x k, with error consistencies between folds on the upper diagonal, and -1
        elsewhere. That is, if we denote one of these matrices as A, then for repeition `r`,  `C[r]`
        = `A[i,j] == 0` if i >= j, and `A[i, j]` is equal to the error consistency between the folds
        i and j, for i, j in {0, 1, ..., k - 1} .

    test_errors: Optional[List[ndarray]] = None
        A list of the boolean error arrays (`y_pred_i != y_test` for fold `i`) for all repetitions.
        Total of `k * repetitions` values if k > 1.

    test_accs: Optional[ndarray] = None
        An array of the accuracies `np.mean(y_pred_i == y_test)` for fold `i` for all repetitions.
        Total of `k * repetitions` values if k > 1.

    test_predictions: Optional[ndarray] = None
        An array of the predictions `y_pred_i` for fold `i` for all repetitions. Total of
        `k * repetitions` values if k > 1.

    fold_accs: Optional[ndarray] = None
        An array of shape `(repetitions, k)` the accuracies (`np.mean(y_pred_fold_i == y_fold_i` for
        fold `i`) for all repetitions. Total of `k * repetitions` values if k > 1.

    fold_predictions: Optional[ndarray] = None
        A NumPy array of shape `(repetitions, k, n_samples)` of the predictions on the *fold* test
        set (`y_pred_fold_i` for fold `i`) for all repetitions.

    fold_models: Optional[ndarray[Model]] = None
        A NumPy object array of size (repetitions, k) where each entry (r, i) is the fitted model on
        repetition `r` fold `i`.
    """

    consistencies: ndarray
    matrix: ndarray
    total_consistency: float
    leave_one_out_consistency: float

    test_errors: Optional[List[ndarray]] = None
    test_accs: Optional[ndarray] = None
    test_predictions: Optional[ndarray] = None

    fold_accs: Optional[ndarray] = None
    fold_predictions: Optional[ndarray] = None
    fold_models: Optional[ndarray] = None


def array_indexer(array: ndarray, sample_dim: int, idx: ndarray) -> ndarray:
    # grotesque but hey, we need to index into a position programmatically
    colons = [":" for _ in range(array.ndim)]
    colons[sample_dim] = "idx"
    idx_string = f"{','.join(colons)}"
    return eval(f"array[{idx_string}]")


def validate_fold(
    x: ndarray,
    y: ndarray,
    x_sample_dim: int,
    y_sample_dim: int,
    train_idx: ndarray,
    val_idx: ndarray,
    model: Model,
    save_fold_accs: bool,
) -> KFoldResults:
    """Perform the internal k-fold validation step on one fold, only doing what is necessary.

    Parameters
    ----------
    train_idx: ndarray
        Fold / validation training indices (see notes above).

    val_idx: ndarray
        Fold / valdiation testing indices (i.e. NOT the final holdout testin indices / set).

    compute_score: bool
        Whether or not to compute an accuracy.

    Returns
    -------
    kfold_results: KFoldResults
        Scroll up in the source code.
    """
    # regardless of options, we need to fit the training set
    if y.ndim == 2:
        raise NotImplementedError("Need to collapse one-hots still")

    x_train = array_indexer(x, x_sample_dim, train_idx)
    y_train = array_indexer(y, y_sample_dim, train_idx)
    model.fit(x_train, y_train)

    acc = None
    y_pred = None
    if save_fold_accs:
        x_val = array_indexer(x, x_sample_dim, val_idx)
        y_val = array_indexer(y, y_sample_dim, val_idx)
        y_pred = model.predict(x_val)
        acc = 1 - np.mean(get_y_error(y_pred, y_val, y_sample_dim))

    return KFoldResults(model, acc, y_pred)


def validate_kfold(
    x: ndarray,
    y: ndarray,
    x_sample_dim: int,
    y_sample_dim: int,
    kfold: Any,
    models: List[Model],
    idx: ndarray,
    save_fold_accs: bool,
) -> List[KFoldResults]:
    """Perform the internal k-fold validation step on one fold, only doing what is necessary.

    Parameters
    ----------
    train_idx: ndarray
        Fold / validation training indices (see notes above).

    val_idx: ndarray
        Fold / valdiation testing indices (i.e. NOT the final holdout testin indices / set).

    compute_score: bool
        Whether or not to compute an accuracy.

    Returns
    -------
    kfold_results: KFoldResults
        Scroll up in the source code.
    """
    results = []
    for model, (train_idx, val_idx) in zip(models, kfold.split(idx)):
        # regardless of options, we need to fit the training set
        if y.ndim == 2:
            raise NotImplementedError("Need to collapse one-hots still")

        x_train = array_indexer(x, x_sample_dim, train_idx)
        y_train = array_indexer(y, y_sample_dim, train_idx)
        model.fit(x_train, y_train)

        acc = None
        y_pred = None
        if save_fold_accs:
            x_val = array_indexer(x, x_sample_dim, val_idx)
            y_val = array_indexer(y, y_sample_dim, val_idx)
            y_pred = model.predict(x_val)
            acc = 1 - np.mean(get_y_error(y_pred, y_val, y_sample_dim))

        results.append(KFoldResults(model, acc, y_pred))
    return results


def validate_kfold_imap(args: Any) -> List[KFoldResults]:
    return validate_kfold(*args)


def get_test_predictions(args: Any) -> List[ndarray]:
    results_list, x_test = args
    y_preds = []
    for results in results_list:
        fitted = results.fitted_model
        y_pred = fitted.predict(x_test)
        y_preds.append(y_pred)
    return y_preds


class ErrorConsistency(ABC):
    """Base class for functionality that all error consistency calculatings must perform.

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

        If None (default), it will be assumed that the `.predict` or `.test` method of the instance
        of `model` takes x as its first positional argument, as in e.g.
        `model.predict(x, **predict_args)`.

        If `predict_args_x` is a string, then a dict will be constructed internally with this
        string, e.g.

            args_dict = {**{predict_args_x: x_train}, **model_args}
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

    Notes
    -----
    Conceptually, for each repetition, there are two steps to computing a k-fold error consistency
    with holdout set:

        (1) evaluation on standard k-fold ("validation" or "folding")
        (2) evaluation on holdout set (outside of k-fold) ("testing")

    There are a lot of overlapping terms and concepts here, so with analogy to deep learning, we
    shall refer to step (1) as *validation* or *val* and step (2) as *testing* or *test*. This will
    help keep variable names and function arguments sane and clear. We refer to the *entire* process
    of validation + testing as *evaluation*. Thus the .evaluate() method with have both validation
    and testing steps, in this terminology.

    Since validation is very much just standard k-fold, we also thus refer to validation steps as
    *fold* steps. So for example validation or fold scores are the k accuracies on the non-training
    partitions of each k-fold repetition (k*repetitions total), but test scores are the
    `repititions` accuracies on the heldout test set.

    The good thing is that standard k-fold is standard k-fold no matter how we implement
    error-consistency (e.g. with holdout, Monte-Carlo style subsetting, etc). We just have train and
    (fold) test indices, and do the usual fit calls and etc. So this can be abstracted to the base
    error consistency class.
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
        empty_unions: str = "drop",
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
        self.empty_unions = empty_unions

        self.x, self.y, self.x_df, self.y_df = self.save_x_y(x, y)
        dim_info = self.save_dims(x, y, x_sample_dim, y_sample_dim)
        self.x_transpose_shape, self.y_transpose_shape = dim_info[:2]
        self.x_sample_dim, self.y_sample_dim = dim_info[2:]

    @staticmethod
    def save_x_y(x: ndarray, y: ndarray) -> Tuple[ndarray, ndarray, PandasData, PandasData]:
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
    def save_dims(
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

    @staticmethod
    def seeds(seed: Optional[int], repetitions: int) -> ndarray:
        MAX_SEED = 2 ** 32 - 1
        if seed is not None:
            random.seed(seed)
            rng = np.random.default_rng(seed)
            return rng.integers(0, MAX_SEED, repetitions)
        rng = np.random.default_rng()
        return rng.integers(0, MAX_SEED, repetitions)

    def array_x_indexer(self, array: ndarray, idx: ndarray) -> ndarray:
        # grotesque but hey, we need to index into a position programmatically
        colons = [":" for _ in range(self.x.ndim)]
        colons[self.x_sample_dim] = "idx"
        idx_string = f"{','.join(colons)}"
        return eval(f"array[{idx_string}]")

    def array_y_indexer(self, array: ndarray, idx: ndarray) -> ndarray:
        # grotesque but hey, we need to index into a position programmatically
        colons = [":" for _ in range(self.y.ndim)]
        colons[self.y_sample_dim] = "idx"
        idx_string = f"{','.join(colons)}"
        return eval(f"array[{idx_string}]")

    def validate_fold(
        self, train_idx: ndarray, val_idx: ndarray, save_fold_accs: bool
    ) -> KFoldResults:
        """Perform the internal k-fold validation step on one fold, only doing what is necessary.

        Parameters
        ----------
        train_idx: ndarray
            Fold / validation training indices (see notes above).

        val_idx: ndarray
            Fold / valdiation testing indices (i.e. NOT the final holdout testin indices / set).

        compute_score: bool
            Whether or not to compute an accuracy.

        Returns
        -------
        kfold_results: KFoldResults
            Scroll up in the source code.
        """
        # regardless of options, we need to fit the training set
        if self.y.ndim == 2:
            raise NotImplementedError("Need to collapse one-hots still")

        x_train = self.array_x_indexer(self.x, train_idx)
        y_train = self.array_y_indexer(self.y, train_idx)
        model = self.model_factory.create()
        model.fit(x_train, y_train)

        acc = None
        y_pred = None
        if save_fold_accs:
            x_val = self.array_x_indexer(self.x, val_idx)
            y_val = self.array_y_indexer(self.y, val_idx)
            y_pred = model.predict(x_val)
            acc = 1 - np.mean(get_y_error(y_pred, y_val, self.y_sample_dim))

        return KFoldResults(model, acc, y_pred)


class ErrorConsistencyKFoldHoldout(ErrorConsistency):
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

        If None (default), it will be assumed that the `.predict` or `.test` method of the instance
        of `model` takes x as its first positional argument, as in e.g.
        `model.predict(x, **predict_args)`.

        If `predict_args_x` is a string, then a dict will be constructed internally with this
        string, e.g.

            args_dict = {**{predict_args_x: x_train}, **model_args}
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

    Notes
    -----
    Conceptually, for each repetition, there are two steps to computing a k-fold error consistency
    with holdout set:

        (1) evaluation on standard k-fold ("validation" or "folding")
        (2) evaluation on holdout set (outside of k-fold) ("testing")

    There are a lot of overlapping terms and concepts here, so with analogy to deep learning, we
    shall refer to step (1) as *validation* or *val* and step (2) as *testing* or *test*. This will
    help keep variable names and function arguments sane and clear. We refer to the *entire* process
    of validation + testing as *evaluation*. Thus the .evaluate() method with have both validation
    and testing steps, in this terminology.

    Since validation is very much just standard k-fold, we also thus refer to validation steps as
    *fold* steps. So for example validation or fold scores are the k accuracies on the non-training
    partitions of each k-fold repetition (k*repetitions total), but test scores are the
    `repititions` accuracies on the heldout test set.
    """

    def __init__(
        self,
        model: Any,
        x: ndarray,
        y: ndarray,
        n_splits: int,
        model_args: Optional[Dict[str, Any]] = None,
        fit_args: Optional[Dict[str, Any]] = None,
        fit_args_x_y: Optional[Tuple[str, str]] = None,
        predict_args: Optional[Dict[str, Any]] = None,
        predict_args_x: Optional[str] = None,
        stratify: bool = False,
        x_sample_dim: int = 0,
        y_sample_dim: int = 0,
        empty_unions: UnionHandling = "zero",
    ) -> None:
        super().__init__(
            model,
            x,
            y,
            n_splits=n_splits,
            model_args=model_args,
            fit_args=fit_args,
            fit_args_x_y=fit_args_x_y,
            predict_args=predict_args,
            predict_args_x=predict_args_x,
            stratify=stratify,
            x_sample_dim=x_sample_dim,
            y_sample_dim=y_sample_dim,
            empty_unions=empty_unions,
        )

    def evaluate(
        self,
        x_test: ndarray,
        y_test: ndarray,
        repetitions: int = 5,
        save_test_accs: bool = True,
        save_test_errors: bool = False,
        save_test_predictions: bool = False,
        save_fold_accs: bool = False,
        save_fold_preds: bool = False,
        save_fold_models: bool = False,
        show_progress: bool = True,
        parallel_reps: bool = False,
        loo_parallel: bool = False,
        turbo: bool = False,
        seed: int = None,
    ) -> ConsistencyResults:
        """Evaluate the error consistency of the classifier.

        Parameters
        ----------
        x_test: Union[List, pandas.DataFrame, pandas.Series, numpy.ndarray]
            ArrayLike object containing holdout predictor samples that the model will never be
            trained or fitted on. Must be have a format identical to that of `x` passed into
            constructor (see above).

        y_test: Union[List, pandas.DataFrame, pandas.Series, numpy.ndarray]
            ArrayLike object containing holdout target values that the model will never be trained
            or fitted on. Must be have a format identical to that of `x` passed into constructor
            (see above).

        repetitions: int = 5
            How many times to repeat the k-fold process. Yields `k*repetitions` error consistencies
            if both `x_test` and `y_test` are proided, and `repetitions*(repititions - 1)/2`
            consistencies otherwise. Note that if both `x_test` and `y_test` are not provided, then
            setting repetitions to 1 will raise an error, since this results in insufficient arrays
            to compare errors.

        save_test_accs: bool = True
            If True (default) also compute accuracy scores for each fold on `x_test` and save them
            in `results.scores`. If False, skip this step. Setting to `False` is useful when
            prediction is expensive and/or you only care about evaulating the error consistency.

        save_test_errors: bool = False
            If True, save a list of the boolean error arrays (`y_pred_i != y_test` for fold `i`) for
            all repetitions in `results.test_errors`. Total of `k * repetitions` values if k > 1.
            If False (default), `results.test_errors` will be `None`.

        save_test_predictions: bool = False
            If True, save an array of the predictions `y_pred_i` for fold `i` for all repetitions in
            `results.test_predictions`. Total of `k * repetitions` values if k > 1. If False
            (default), `results.test_predictions` will be `None`.

        save_fold_accs: bool = False
            If True, save an array of shape `(repetitions, k)` of the predictions on the *fold* test
            set (`y_pred_fold_i` for fold `i`) for all repetitions in `results.fold_accs`.

        save_fold_preds: bool = False
            If True, save a NumPy array of shape `(repetitions, k, n_samples)` of the predictions on
            the *fold* test set (`y_pred_fold_i` for fold `i`) for all repetitions in
            `results.fold_predictions`.

        save_fold_models: bool = False
            If True, `results.fold_models` is a NumPy object array of size (repetitions, k) where
            each entry (r, i) is the fitted model on repetition `r` fold `i`.

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

        self.x_test, self.y_test = self.save_x_y(x_test, y_test)[0:2]
        seeds = self.seeds(seed, repetitions)

        kfolds = [KFold(n_splits=self.n_splits, shuffle=True, random_state=seed) for seed in seeds]

        test_errors: List[ndarray] = []
        test_accs: ndarray = []
        test_predictions: ndarray = []

        fold_accs: ndarray = []
        fold_predictions: ndarray = []
        fold_models: ndarray = []
        idx = np.arange(0, int(self.x.shape[self.x_sample_dim]), dtype=int)
        if not parallel_reps:
            rep_desc, fold_desc = "K-fold Repetition {}", "Fold {}"
            rep_pbar = tqdm(total=repetitions, desc=rep_desc.format(0), leave=True)
            for rep, kfold in enumerate(kfolds):  # we have `repetitions` ("rep") kfold partitions
                rep_pbar.set_description(rep_desc.format(rep))
                fold_pbar = tqdm(total=self.n_splits, desc=fold_desc.format(0), leave=False)
                for k, (train_idx, test_idx) in enumerate(kfold.split(idx)):
                    fold_pbar.set_description(fold_desc.format(k))
                    results = self.validate_fold(train_idx, test_idx, save_fold_accs)
                    if save_fold_accs:
                        fold_accs.append(results.score)
                    if save_fold_preds:
                        fold_predictions.append(results.prediction)
                    if save_fold_models:
                        fold_models.append(results.fitted_model)
                    fitted = results.fitted_model
                    y_pred = fitted.predict(x_test)
                    y_err = get_y_error(y_pred, y_test, self.y_sample_dim)
                    acc = 1 - np.mean(y_err)
                    test_predictions.append(y_pred)
                    test_accs.append(acc)
                    test_errors.append(y_err)
                    fold_pbar.update()
                fold_pbar.close()
                rep_pbar.update()
            rep_pbar.close()
        else:
            # Yes, this is all very grotesque
            # fmt: off
            models_list = (
                [self.model_factory.create() for _ in range(self.n_splits)] for _ in range(repetitions)
            )
            args = (
                (
                    self.x, self.y, self.x_sample_dim, self.y_sample_dim,
                    kfold, models, idx, save_fold_accs,
                )
                for kfold, models in zip(kfolds, models_list)
            )
            # fmt: on
            # with Pool(processes=cpu_count()) as pool:
            #     rep_results = pool.starmap(validate_kfold, args)

            rep_results = process_map(
                validate_kfold_imap,
                args,
                max_workers=cpu_count(),
                desc="Repeating k-fold",
                total=repetitions,
            )

            predict_args = [(rep_result, x_test) for rep_result in rep_results]

            y_preds_list = process_map(
                get_test_predictions,
                predict_args,
                max_workers=cpu_count(),
                desc="Computing holdout predictions",
                total=repetitions,
            )

            # for results_list in tqdm(rep_results, desc="Computing test predictions", total=repetitions):
            #     for results in results_list:
            #         fitted = results.fitted_model
            #         y_pred = fitted.predict(x_test)

            for results_list, y_preds in tqdm(
                zip(rep_results, y_preds_list), desc="Saving results", total=repetitions
            ):
                for results, y_pred in zip(results_list, y_preds):
                    if save_fold_accs:
                        fold_accs.append(results.score)
                    if save_fold_preds:
                        fold_predictions.append(results.prediction)
                    if save_fold_models:
                        fold_models.append(results.fitted_model)
                    fitted = results.fitted_model
                    y_err = get_y_error(y_pred, y_test, self.y_sample_dim)
                    acc = 1 - np.mean(y_err)
                    test_predictions.append(y_pred)
                    test_accs.append(acc)
                    test_errors.append(y_err)

        print("Computing consistencies")
        errcon_results = error_consistencies(
            test_predictions,
            y_test,
            self.y_sample_dim,
            empty_unions=self.empty_unions,
            loo_parallel=loo_parallel,
            turbo=turbo,
        )

        consistencies, matrix, unpredictables, predictables, loo_consistencies = errcon_results
        numerator = np.sum(unpredictables)
        total = numerator / np.sum(predictables) if numerator > 0 else 0

        return ConsistencyResults(
            consistencies=consistencies,
            matrix=matrix,
            total_consistency=total,
            leave_one_out_consistency=np.mean(loo_consistencies),
            test_errors=test_errors if save_test_errors else None,
            test_accs=test_accs if save_fold_accs else None,
            test_predictions=test_predictions if save_test_predictions else None,
            fold_accs=fold_accs if save_fold_accs else None,
            fold_predictions=fold_predictions if save_fold_preds else None,
            fold_models=fold_models if save_fold_models else None,
        )


class ErrorConsistencyInternalKFold(ErrorConsistency):
    def __init__(
        self,
        model: Any,
        x: ndarray,
        y: ndarray,
        n_splits: int,
        model_args: Optional[Dict[str, Any]],
        fit_args: Optional[Dict[str, Any]],
        fit_args_x_y: Optional[Tuple[str, str]],
        predict_args: Optional[Dict[str, Any]],
        predict_args_x: Optional[str],
        stratify: bool,
        x_sample_dim: int,
        y_sample_dim: int,
    ) -> None:
        super().__init__(
            model,
            x,
            y,
            n_splits=n_splits,
            model_args=model_args,
            fit_args=fit_args,
            fit_args_x_y=fit_args_x_y,
            predict_args=predict_args,
            predict_args_x=predict_args_x,
            stratify=stratify,
            x_sample_dim=x_sample_dim,
            y_sample_dim=y_sample_dim,
        )

    def evaluate(
        self,
        repetitions: int,
        compute_scores: bool,
        save_error_arrays: bool,
        save_predictions: bool,
        save_kfold_predictions: bool,
        save_models: bool,
        seed: int,
    ) -> ConsistencyResults:
        """Evaluate the error consistency of the classifier.

        Parameters
        ----------
        repetitions: int = 5
            How many times to repeat the k-fold process. Yields `repetitions*(repititions - 1)/2`
            consistencies if `repetitions` is greater than 1. Setting repetitions to 1 instead uses
            the entire set `X` for prediction for each fold, thus yield `k*(k-1)/2` consistencies,
            but which are strongly biased toward a value much lower than the true consistency.
            Useful for quick checks / fast estimates of upper bounds on the error consistency, but
            otherwise not recommended.

        compute_scores: bool = True
            If True (default) also compute accuracy scores for each fold and save them in
            `results.scores`. If False, `x_test` and `y_test` are not None, only compute predictions
            on `x_test`, and not testing subsets of each k-fold partition. Useful if prediction is
            expensive or otherwise not needed.

        save_error_arrays: bool = False,
            If True also save the boolean arrays indicating the error locations of each fold in
            `results.error_arrays`. If False (default), leave this property empty in the results.
            Note if `x_test` and `y_test` are provided, the error arrays are on these values, but if
            `x_test` and `y_test` are not provided, error arrays are on the k-fold predictions.

        save_predictions: bool = False
            If True and `x_test` and `y_test` are not None, also save the predicted target values on
            `x_test` for each each fold in `results.val_predictions`. If False (default), leave this
            property empty in the results.

        save_kfold_predictions: bool = False
            If True also save the predicted target values of each internal k-fold in
            `results.fold_predictions`. If False (default), leave this property empty in the
            results.

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
        pass


class ErrorConsistencyMonteCarlo:
    """Calculate error consistency using repeated random train/test splits."""
