from __future__ import annotations

from abc import ABC
from multiprocessing import cpu_count
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from error_consistency.containers import ConsistencyResults, KFoldResults
from error_consistency.functional import UnionHandling, error_consistencies, get_y_error
from error_consistency.model import Model, ModelFactory
from error_consistency.parallel import (
    get_test_predictions,
    get_test_predictions_internal,
    validate_kfold_imap,
)
from error_consistency.random import parallel_seed_generators, random_seeds
from error_consistency.utils import array_indexer, to_numpy

PandasData = Union[DataFrame, Series]
Shape = Tuple[int, ...]


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

    :meta private:
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

    return KFoldResults(model, acc, y_pred, val_idx)


class ErrorConsistencyBase(ABC):
    """Base class for functionality that all error consistency calculatings must perform.

    Parameters
    ----------
    model: Intersection[Callable, Type]
        A *class* where instances are classifiers that implement:

        1. A ``.fit`` or ``.train`` method that:

           #. accepts predictors and targets, plus `fit_args`, and
           #. updates the state of `model` when calling `.fit` or `.train`

        2. A ``.predict`` or ``.test`` method, that:

           #. accepts testing samples, plus ``predict_args``, and
           #. requires having called ``.fit`` previously, and
           #. returns *only* the predictions as a single ArrayLike (e.g. NumPy array, List, pandas
              DataFrame or Series)

        E.g.::

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

    :meta private:
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
        empty_unions: UnionHandling = 0,
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
        """ :meta private: """
        x_df = x if isinstance(x, DataFrame) or isinstance(x, Series) else None
        y_df = y if isinstance(y, DataFrame) or isinstance(y, Series) else None
        x = to_numpy(x)
        y = to_numpy(y)
        if y.ndim > 2:
            raise ValueError("Target `y` can only be 1-dimensional or two-dimensional.")
        if y.ndim == 2:
            uniques = np.unique(y.ravel()).astype(int)
            if not np.array_equal(uniques, [0, 1]):
                raise ValueError("Only dummy-coded and one-hot coded 2D targets are supported.")

        return x, y, x_df, y_df

    @staticmethod
    def save_dims(
        x: ndarray, y: ndarray, x_sample_dim: int, y_sample_dim: int
    ) -> Tuple[Shape, Shape, int, int]:
        """ :meta private: """
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

        :meta private:
        """
        # regardless of options, we need to fit the training set
        if self.y.ndim == 2:
            raise NotImplementedError("Need to collapse one-hots still")

        x_train = array_indexer(self.x, self.x_sample_dim, train_idx)
        y_train = array_indexer(self.y, self.y_sample_dim, train_idx)
        model = self.model_factory.create()
        model.fit(x_train, y_train)

        acc = None
        y_pred = None
        if save_fold_accs:
            x_val = array_indexer(self.x, self.x_sample_dim, val_idx)
            y_val = array_indexer(self.y, self.y_sample_dim, val_idx)
            y_pred = model.predict(x_val)
            acc = 1 - np.mean(get_y_error(y_pred, y_val, self.y_sample_dim))

        return KFoldResults(model, acc, y_pred, val_idx)

    def starmap_args(
        self, repetitions: int, save_fold_accs: bool, seed: Optional[int]
    ) -> Generator[Tuple[Any, ...], None, None]:
        rngs = parallel_seed_generators(seed, repetitions)
        model = self.model_factory.create
        for i in range(repetitions):
            yield (
                self.x,
                self.y,
                self.x_sample_dim,
                self.y_sample_dim,
                self.n_splits,
                self.stratify,
                [model() for _ in range(self.n_splits)],
                rngs[i],
                save_fold_accs,
            )


class ErrorConsistencyKFoldHoldout(ErrorConsistencyBase):
    """Compute error consistencies for a classifier.

    Parameters
    ----------
    model: Intersection[Callable, Type]
        A *class* where instances are classifiers that implement:

        1. A ``.fit`` or ``.train`` method that:

           * accepts predictors and targets, plus `fit_args`, and
           * updates the state of `model` when calling `.fit` or `.train`

        2. A ``.predict`` or ``.test`` method, that:

           * accepts testing samples, plus ``predict_args``, and
           * requires having called ``.fit`` previously, and
           * returns *only* the predictions as a single ArrayLike (e.g. NumPy array, List, pandas
             DataFrame or Series)

        .. _valid model example:

        E.g.::

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
        How many folds to use, and thus models to generate, per repetition.

    model_args: Optional[Dict[str, Any]]
        Any arguments that are required each time to construct a fresh instance of the model (see
        the `valid model example`_ above). Note that the data x and y must NOT be included here.

    fit_args: Optional[Dict[str, Any]]
        Any arguments that are required each time when calling the `.fit` or `.train` methods
        internally (see the `valid model example`_ above). Note that the data x and y must NOT be
        included here.

    fit_args_x_y: Optional[Tuple[str, str]] = None
        Name of the arguments which data `x` and target `y` are passed to. This is needed because
        different libraries may have different conventions for how they expect predictors and
        targets to be passed in to `fit` or `train`. For example, a function may have the
        signature::

            f(predictor: ndarray, target: ndarray) -> Any

        To allow our internal `x_train` and `x_test` splits to be passed to the right arguments,
        we thus need to know these names.

        If None (default), it will be assumed that the `.fit` or `.train` method of the instance of
        `model` takes x as its first positional argument, and `y` as its second, as in e.g.
        `model.fit(x, y, **model_args)`.

        If a tuple of strings (x_name, y_name), then a dict will be constructed internally by
        splatting, e.g.::

            args_dict = {**{x_name: x_train, y_name: y_train}, **model_args}
            model.fit(**args_dict)

        Alternately, see the documentation for `error_consistency.model.Model` for how to subclass
        your own function here if you require more fine-grained control of how arguments are passed
        into the fit and predict calls.

    predict_args: Optional[Dict[str, Any]]
        Any arguments that are required each time when calling the `.predict` or `.test` methods
        internally (see the `valid model example`_ above). Note that the data x must NOT be included
        here.

    predict_args_x: Optional[str] = None
        Name of the argument which data `x` is passed to during evaluation. This is needed because
        different libraries may have different conventions for how they expect predictors and
        targets to be passed in to `predict` or `test` calls.

        If None (default), it will be assumed that the `.predict` or `.test` method of the instance
        of `model` takes x as its first positional argument, as in e.g.
        `model.predict(x, **predict_args)`.

        If `predict_args_x` is a string, then a dict will be constructed internally with this
        string, e.g.::

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

    empty_unions: UnionHandling = 0
        When computing the pairwise consistency or leave-one-out consistency on small or
        simple datasets, it can be the case that the union of the error sets is empty (e.g. if no
        prediction errors are made). In this case the intersection over union is 0/0, which is
        undefined.

        * If `0` (default), the consistency for that collection of error sets is set to zero.
        * If `1`, the consistency for that collection of error sets is set to one.
        * If "nan", the consistency for that collection of error sets is set to `np.nan`.
        * If "drop", the `consistencies` array will not include results for that collection,
          but the consistency matrix will include `np.nans`.
        * If "error", an empty union will cause a `ZeroDivisionError`.
        * If "warn", an empty union will print a warning (probably a lot).


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
        empty_unions: UnionHandling = 0,
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
        empty_unions: UnionHandling = 0,
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

        seed: int = None
            Seed for reproducible results

        Returns
        -------
        results: ConsistencyResults
            An `error_consistency.containers.ConsistencyResults` object.
        """

        self.x_test, self.y_test = self.save_x_y(x_test, y_test)[0:2]
        seeds = random_seeds(seed, repetitions)

        if self.stratify:
            kfolds = [
                StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
                for seed in seeds
            ]
        else:
            kfolds = [
                KFold(n_splits=self.n_splits, shuffle=True, random_state=seed) for seed in seeds
            ]

        test_errors: List[ndarray] = []
        test_accs: ndarray = []
        test_predictions: ndarray = []

        fold_accs: ndarray = []
        fold_predictions: ndarray = []
        fold_models: ndarray = []
        idx = np.arange(0, int(self.x.shape[self.x_sample_dim]), dtype=int)
        if self.y.ndim == 2:
            y_split = np.argmax(self.y, axis=1 - np.abs(self.y_sample_dim))  # convert to labels
        else:
            y_split = self.y
        if not parallel_reps:
            rep_desc, fold_desc = "K-fold Repetition {}", "Fold {}"
            rep_pbar = tqdm(total=repetitions, desc=rep_desc.format(0), leave=True)
            for rep, kfold in enumerate(kfolds):  # we have `repetitions` ("rep") kfold partitions
                rep_pbar.set_description(rep_desc.format(rep))
                fold_pbar = tqdm(total=self.n_splits, desc=fold_desc.format(0), leave=False)
                for k, (train_idx, test_idx) in enumerate(kfold.split(idx, y_split)):
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
                    test_predictions.append(y_pred)
                    if save_test_accs:
                        acc = 1 - np.mean(y_err)
                        test_accs.append(acc)
                    test_errors.append(y_err)
                    fold_pbar.update()
                fold_pbar.close()
                rep_pbar.update()
            rep_pbar.close()
        else:
            rep_results = process_map(
                validate_kfold_imap,
                self.starmap_args(repetitions, save_fold_accs, seed),
                max_workers=cpu_count(),
                desc="Repeating k-fold",
                total=repetitions,
            )

            y_preds_list = process_map(
                get_test_predictions,
                [(rep_result, x_test) for rep_result in rep_results],
                max_workers=cpu_count(),
                desc="Computing holdout predictions",
                total=repetitions,
            )

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
                    test_predictions.append(y_pred)
                    if save_test_accs:
                        acc = 1 - np.mean(y_err)
                        test_accs.append(acc)
                    test_errors.append(y_err)

        errcon_results = error_consistencies(
            y_preds=test_predictions,
            y_true=y_test,
            sample_dim=self.y_sample_dim,
            empty_unions=empty_unions,
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
            test_accs=np.array(test_accs) if save_test_accs else None,
            test_predictions=test_predictions if save_test_predictions else None,
            fold_accs=np.array(fold_accs) if save_fold_accs else None,
            fold_predictions=fold_predictions if save_fold_preds else None,
            fold_models=fold_models if save_fold_models else None,
        )

    def starmap_args(
        self, repetitions: int, save_fold_accs: bool, seed: Optional[int]
    ) -> Generator[Tuple[Any, ...], None, None]:
        rngs = parallel_seed_generators(seed, repetitions)
        model = self.model_factory.create
        for i in range(repetitions):
            yield (
                self.x,
                self.y,
                self.x_sample_dim,
                self.y_sample_dim,
                self.n_splits,
                self.stratify,
                [model() for _ in range(self.n_splits)],
                rngs[i],
                save_fold_accs,
            )


class ErrorConsistencyKFoldInternal(ErrorConsistencyBase):
    """Compute error consistencies for a classifier.

    Parameters
    ----------
    model: Intersection[Callable, Type]
        A *class* where instances are classifiers that implement:

        1. A ``.fit`` or ``.train`` method that:

           * accepts predictors and targets, plus `fit_args`, and
           * updates the state of `model` when calling `.fit` or `.train`

        2. A ``.predict`` or ``.test`` method, that:

           * accepts testing samples, plus ``predict_args``, and
           * requires having called ``.fit`` previously, and
           * returns *only* the predictions as a single ArrayLike (e.g. NumPy array, List, pandas
             DataFrame or Series)

        .. _valid model example:

        E.g.::

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
        How many folds to use, and thus models to generate, per repetition.

    model_args: Optional[Dict[str, Any]]
        Any arguments that are required each time to construct a fresh instance of the model (see
        the `valid model example`_ above). Note that the data x and y must NOT be included here.

    fit_args: Optional[Dict[str, Any]]
        Any arguments that are required each time when calling the `.fit` or `.train` methods
        internally (see the `valid model example`_ above). Note that the data x and y must NOT be
        included here.

    fit_args_x_y: Optional[Tuple[str, str]] = None
        Name of the arguments which data `x` and target `y` are passed to. This is needed because
        different libraries may have different conventions for how they expect predictors and
        targets to be passed in to `fit` or `train`. For example, a function may have the
        signature::

            f(predictor: ndarray, target: ndarray) -> Any

        To allow our internal `x_train` and `x_test` splits to be passed to the right arguments,
        we thus need to know these names.

        If None (default), it will be assumed that the `.fit` or `.train` method of the instance of
        `model` takes x as its first positional argument, and `y` as its second, as in e.g.
        `model.fit(x, y, **model_args)`.

        If a tuple of strings (x_name, y_name), then a dict will be constructed internally by
        splatting, e.g.::

            args_dict = {**{x_name: x_train, y_name: y_train}, **model_args}
            model.fit(**args_dict)

        Alternately, see the documentation for `error_consistency.model.Model` for how to subclass
        your own function here if you require more fine-grained control of how arguments are passed
        into the fit and predict calls.

    predict_args: Optional[Dict[str, Any]]
        Any arguments that are required each time when calling the `.predict` or `.test` methods
        internally (see the `valid model example`_ above). Note that the data x must NOT be included
        here.

    predict_args_x: Optional[str] = None
        Name of the argument which data `x` is passed to during evaluation. This is needed because
        different libraries may have different conventions for how they expect predictors and
        targets to be passed in to `predict` or `test` calls.

        If None (default), it will be assumed that the `.predict` or `.test` method of the instance
        of `model` takes x as its first positional argument, as in e.g.
        `model.predict(x, **predict_args)`.

        If `predict_args_x` is a string, then a dict will be constructed internally with this
        string, e.g.::

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

    empty_unions: UnionHandling = 0
        When computing the pairwise consistency or leave-one-out consistency on small or
        simple datasets, it can be the case that the union of the error sets is empty (e.g. if no
        prediction errors are made). In this case the intersection over union is 0/0, which is
        undefined.

        * If `0` (default), the consistency for that collection of error sets is set to zero.
        * If `1`, the consistency for that collection of error sets is set to one.
        * If "nan", the consistency for that collection of error sets is set to `np.nan`.
        * If "drop", the `consistencies` array will not include results for that collection,
          but the consistency matrix will include `np.nans`.
        * If "error", an empty union will cause a `ZeroDivisionError`.
        * If "warn", an empty union will print a warning (probably a lot).


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
        stratify: bool = True,
        x_sample_dim: int = 0,
        y_sample_dim: int = 0,
        empty_unions: UnionHandling = 0,
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
        repetitions: int = 5
            How many times to repeat the k-fold process. Yields `repetitions*(repititions - 1)/2`
            consistencies if `repetitions` is greater than 1. Setting repetitions to 1 instead uses
            the entire set `X` for prediction for each fold, thus yield `k*(k-1)/2` consistencies,
            but which are strongly biased toward a value much lower than the true consistency.
            Useful for quick checks / fast estimates of upper bounds on the error consistency, but
            otherwise not recommended.

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

        seed: int = None
            Seed for reproducible results

        Returns
        -------
        results: ConsistencyResults
            The `error_consistency.containers.ConsistencyResults` object.
        """
        seeds = random_seeds(seed, repetitions)

        if self.stratify:
            kfolds = [
                StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
                for seed in seeds
            ]
        else:
            kfolds = [
                KFold(n_splits=self.n_splits, shuffle=True, random_state=seed) for seed in seeds
            ]

        test_errors: List[ndarray] = []
        test_accs: ndarray = []
        test_predictions: ndarray = []

        fold_accs: ndarray = []
        fold_predictions: ndarray = []
        fold_models: ndarray = []
        idx = np.arange(0, int(self.x.shape[self.x_sample_dim]), dtype=int)
        if self.y.ndim == 2:
            y_split = np.argmax(self.y, axis=1 - np.abs(self.y_sample_dim))  # convert to labels
        else:
            y_split = self.y
        if not parallel_reps:
            rep_desc, fold_desc = "K-fold Repetition {}", "Fold {}"
            rep_pbar = tqdm(total=repetitions, desc=rep_desc.format(0), leave=True)
            for rep, kfold in enumerate(kfolds):  # we have `repetitions` ("rep") kfold partitions
                fold_combined_preds = np.full_like(y_split, -1)
                rep_pbar.set_description(rep_desc.format(rep))
                fold_pbar = tqdm(total=self.n_splits, desc=fold_desc.format(0), leave=False)

                for k, (train_idx, test_idx) in enumerate(kfold.split(idx, y_split)):
                    fold_pbar.set_description(fold_desc.format(k))
                    # We have `save_fold_accs=True` below because we need to run the predictions
                    # to assemble the piecewise predictions, regardless of whether or not we save
                    # the piecewise predictions individually later
                    results = self.validate_fold(train_idx, test_idx, save_fold_accs=True)
                    y_pred = results.prediction
                    fold_combined_preds[test_idx] = y_pred
                    if save_fold_accs:
                        fold_accs.append(results.score)
                    if save_fold_preds:
                        fold_predictions.append(y_pred)
                    if save_fold_models:
                        fold_models.append(results.fitted_model)
                    fold_pbar.update()
                fold_pbar.close()

                y_pred = fold_combined_preds
                y_err = get_y_error(y_pred, y_split, self.y_sample_dim)
                acc = 1 - np.mean(y_err)
                test_predictions.append(y_pred)
                test_accs.append(acc)
                test_errors.append(y_err)
                rep_pbar.update()
            rep_pbar.close()
        else:
            rep_results: List[List[KFoldResults]] = process_map(
                validate_kfold_imap,
                # We have `save_fold_accs=True` (repetitions, True, seed) below because we need to
                # run the predictions to assemble the piecewise predictions, regardless of whether
                # or not we save the piecewise predictions individually later
                self.starmap_args(repetitions, True, seed),
                max_workers=cpu_count(),
                desc="Repeating k-fold",
                total=repetitions,
            )
            results_list: List[KFoldResults]
            for results_list in tqdm(rep_results, desc="Saving results", total=repetitions):
                fold_combined_preds = np.full_like(y_split, -1)
                for results in results_list:
                    y_pred = results.prediction
                    test_idx = results.test_idx
                    fold_combined_preds[test_idx] = y_pred
                    if save_fold_accs:
                        fold_accs.append(results.score)
                    if save_fold_preds:
                        fold_predictions.append(y_pred)
                    if save_fold_models:
                        fold_models.append(results.fitted_model)
                y_err = get_y_error(fold_combined_preds, y_split, self.y_sample_dim)
                acc = 1 - np.mean(y_err)
                test_predictions.append(fold_combined_preds)
                test_accs.append(acc)
                test_errors.append(y_err)

        errcon_results = error_consistencies(
            y_preds=test_predictions,
            y_true=y_split,
            sample_dim=self.y_sample_dim,
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
            test_accs=np.array(test_accs) if save_test_accs else None,
            test_predictions=test_predictions if save_test_predictions else None,
            fold_accs=np.array(fold_accs) if save_fold_accs else None,
            fold_predictions=fold_predictions if save_fold_preds else None,
            fold_models=fold_models if save_fold_models else None,
        )


class ErrorConsistencyMonteCarlo:
    """Calculate error consistency using repeated random train/test splits."""


class ErrorConsistency(ErrorConsistencyBase):
    """Compute the error consistency of a classifier.


    Parameters
    ----------
    model: Intersection[Callable, Type]
        A *class* where instances are classifiers that implement:

        1. A ``.fit`` or ``.train`` method that:

           #. accepts predictors and targets, plus `fit_args`, and
           #. updates the state of `model` when calling `.fit` or `.train`

        2. A ``.predict`` or ``.test`` method, that:

           #. accepts testing samples, plus ``predict_args``, and
           #. requires having called ``.fit`` previously, and
           #. returns *only* the predictions as a single ArrayLike (e.g. NumPy array, List, pandas
              DataFrame or Series)

        E.g.::

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
        `model.fit(x, y, **model_args)` for arguments `model` and `model_args`. If using external
        validation (e.g. passing `x_test` into `ErrorConsistency.evaluate`), you must ensure `x`
        does not contain `x_test`, that is, this argument functions as if it is `x_train`.

        Otherwise, if using internal validation, splitting of x into validation subsets will be
        along the first axis (axis 0), that is, the first axis is assumed to be the sample
        dimension. If your fit method requires a different sample dimension, you can specify this
        in `x_sample_dim`.

    y: Union[List, pandas.DataFrame, pandas.Series, numpy.ndarray]
        ArrayLike object containing targets. Must be in a format that is consumable with
        `model.fit(x, y, **model_args)` for arguments `model` and `model_args`. If using external
        validation (e.g. passing `x_test` into `ErrorConsistency.evaluate`), you must ensure `x`
        does not contain `x_test`, that is, this argument functions as if it is `x_train`.

        Otherwise, if using internal validation, splitting of y into validation subsets will be
        along the first axis (axis 0), that is, the first axis is assumed to be the sample
        dimension. If your fit method requires a different sample dimension (e.g. y is a one-hot
        encoded array), you can specify this in `y_sample_dim`.

    n_splits: int = 5
        How many folds to use for validating error consistency. Only relevant

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

    empty_unions: UnionHandling = 0
        When computing the pairwise consistency or leave-one-out consistency on small or
        simple datasets, it can be the case that the union of the error sets is empty (e.g. if no
        prediction errors are made). In this case the intersection over union is 0/0, which is
        undefined.

        * If `0` (default), the consistency for that collection of error sets is set to zero.
        * If `1`, the consistency for that collection of error sets is set to one.
        * If "nan", the consistency for that collection of error sets is set to `np.nan`.
        * If "drop", the `consistencies` array will not include results for that collection,
          but the consistency matrix will include `np.nans`.
        * If "error", an empty union will cause a `ZeroDivisionError`.
        * If "warn", an empty union will print a warning (probably a lot).

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

    :meta public:
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
        empty_unions: UnionHandling = 0,
        onehot_y: bool = True,
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
        repetitions: int = 5,
        x_test: ndarray = None,
        y_test: ndarray = None,
        save_test_accs: bool = True,
        save_test_errors: bool = False,
        save_test_predictions: bool = False,
        save_fold_accs: bool = False,
        save_fold_preds: bool = False,
        save_fold_models: bool = False,
        empty_unions: UnionHandling = 0,
        show_progress: bool = True,
        parallel_reps: bool = False,
        loo_parallel: bool = False,
        turbo: bool = False,
        seed: int = None,
    ) -> ConsistencyResults:
        """Evaluate the error consistency of the classifier.

        Parameters
        ----------
        repetitions: int = 5
            How many times to repeat the k-fold process. Yields `k*repetitions` error consistencies
            if both `x_test` and `y_test` are provided, and `repetitions*(repititions - 1)/2`
            consistencies otherwise. Note that if both `x_test` and `y_test` are not provided, then
            setting repetitions to 1 will raise an error, since this results in insufficient arrays
            to compare errors.

        x_test: Union[List, pandas.DataFrame, pandas.Series, numpy.ndarray]
            ArrayLike object containing holdout predictor samples that the model will never be
            trained or fitted on. Must be have a format identical to that of `x` passed into
            constructor (see above).

        y_test: Union[List, pandas.DataFrame, pandas.Series, numpy.ndarray]
            ArrayLike object containing holdout target values that the model will never be trained
            or fitted on. Must be have a format identical to that of `x` passed into constructor
            (see above).

        save_test_accs: bool = True
            If True (default) also compute accuracy scores and save them in the returned
            `results.scores`. If False, skip this step.

            Note: when `x_test` and `y_test` are provided, test accuracies are over these values.
            When not provided, test accuracies are over the entire set `y` passed into the
            `ErrorConsistency` constructor, but constructed from each fold (e.g. if there are `k`
            splits, the predictions on the k disjoint folds are joined together to get one total
            set of predictions for that repetition).

        save_test_errors: bool = False
            If True, save a list of the boolean error arrays (`y_pred != y_test`) for all
            repetitions. If False (default), the return value `results` will have
            `results.test_errors` be `None`.

            Note: when `x_test` and `y_test` are provided, errors are on `y_test`.
            When not provided, test accuracies are over the entire set `y` passed into the
            `ErrorConsistency` constructor, but constructed from each fold (e.g. if there are `k`
            splits, the predictions on the k disjoint folds are joined together to get one total
            set of predictions for that repetition).

        save_test_predictions: bool = False
            If True, save an array of the predictions `y_pred_i` for fold `i` for all repetitions in
            `results.test_predictions`. Total of `k * repetitions` values if k > 1. If False
            (default), `results.test_predictions` will be `None`.

            Note: when `x_test` and `y_test` are provided, predictions are for `y_test`.
            When not provided, predictions are for the entire set `y` passed into the
            `error_consistency.consistency.ErrorConsistency` constructor, but constructed from the
            models trained on each disjoint fold (e.g. if there are `k` splits, the predictions on
            the `k` disjoint folds are joined together to get one total set of predictions for that
            repetition). That is, the predictions are the combined results of `k` different models.

        save_fold_accs: bool = False
            If True, save a list of shape `(repetitions, k)` of the predictions on the *fold* test
            sets for all repetitions. This list will be available in `results.fold_accs`. If False,
            do not save these values.

            Note: when `x_test` and `y_test` are provided, and `save_fold_accs=False` and
            `save_fold_preds=False`, then the entire prediction and accuracy evaluation on each
            k-fold will be skipped, potentially saving significant compute time, depending on the
            model and size of the dataset. However, when using an internal validation method
            (`x_test` and `y_test` are not provided) this prediction step still must be executed.

        save_fold_preds: bool = False
            If True, save a list with shape `(repetitions, k, n_samples)` of the predictions on
            the *fold* test set for all repetitions. This list will be abalable in
            `results.fold_predictions`. If False, do not save these values. See Notes above for
            extra details on this behaviour.

        save_fold_models: bool = False
            If True, `results.fold_models` is a nested list of size (repetitions, k) where
            each entry (r, i) is the *fitted* model on repetition `r` fold `i`.

            Note: During parallelization, new models are constructed each time using the passed in
            `model` class and the model arguments.Parallelization pickles these models and the
            associated data, and then the actual models are fit in each separate process. When
            there is no parallelization, the procedure is still similar, in that separate models
            are created for every repetition. Thus, you have to be careful about memory when using
            `save_fold_models` and a large number of repetions. The `error-consistency` library
            wraps all `model` classes passed in into a `Model` class which is used internally to
            unify interfacing across various libraries. This `Model` class is very tiny, and is not
            a concern for memory, but if the wrapped model is large, you may have memory problems.
            E.g. KNN and other memory-based methods which may have an option `save_x_y` or the like
            could lead to problems when using `save_fold_models=True`.

        seed: int = None
            Seed for reproducible results.

        Returns
        -------
        results: ConsistencyResults
            An `error_consistency.containers.ConsistencyResults` object.
        """
        if (x_test, y_test) == (None, None):
            self.consistency_class: Type[ErrorConsistencyBase] = ErrorConsistencyKFoldHoldout
        elif (x_test is not None) and (y_test is not None):
            self.consistency_class = ErrorConsistencyKFoldInternal
        else:
            raise ValueError(
                "If providing external holdout data, *both* `x_test` and `y_test` must be provided."
            )

