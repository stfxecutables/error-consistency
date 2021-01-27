from __future__ import annotations

from typing import Any, List

import numpy as np
from numpy import ndarray
from numpy.random import Generator as RandomGenerator
from sklearn.model_selection import KFold, StratifiedKFold

from error_consistency.containers import KFoldResults
from error_consistency.functional import get_y_error
from error_consistency.model import Model
from error_consistency.utils import array_indexer


def validate_kfold(
    x: ndarray,
    y: ndarray,
    x_sample_dim: int,
    y_sample_dim: int,
    n_splits: int,
    stratify: bool,
    models: List[Model],
    generator: RandomGenerator,
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

    :meta private:
    """
    # don't work with original arrays which may have sample index in strange location
    idx = np.arange(0, int(x.shape[x_sample_dim]), dtype=int)
    # convert to labels if we have a one-hot (will fail for dummy coding)
    y_split = np.argmax(y, axis=1 - int(np.abs(y_sample_dim))) if y.ndim == 2 else y

    # can't trust kfold shuffling in multiprocessing, shuffle ourselves
    generator.shuffle(idx)
    y_split = y_split[idx]

    kfolder = StratifiedKFold if stratify else KFold
    kfold = kfolder(n_splits=n_splits, shuffle=False)

    results = []
    for model, (train_idx, val_idx) in zip(models, kfold.split(idx, y_split)):
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
    """:meta private:"""
    return validate_kfold(*args)


def get_test_predictions(args: Any) -> List[ndarray]:
    """:meta private:"""
    results_list, x_test = args
    y_preds = []
    for results in results_list:
        fitted = results.fitted_model
        y_pred = fitted.predict(x_test)
        y_preds.append(y_pred)
    return y_preds
