from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from numpy import ndarray
from pandas import Series, DataFrame
import numpy as np


from typing import List, Optional

from error_consistency.model import Model


@dataclass(eq=False)
class KFoldResults:
    """Hold results of a single fold.

    :meta private:

    Attributes
    ----------
    fitted_model: Model
        The fitted `error_consistency.model.Model`.

    score: Optional[float]
        The accuracy on the internal k-fold test set.

    prediction: ndarray
        The predicted target values on the internal k-fold test set.
    """

    fitted_model: Model
    score: Optional[ndarray]
    prediction: ndarray
    test_idx: ndarray


@dataclass(eq=False)
class ConsistencyResults:
    """Holds results from evaluating error consistency.

    Attributes
    ----------
    consistencies: ndarray
        A flat array of all the pairwise consistencies. Length will be N*(N-1)/2, where for `n_rep`
        reptitions of k-fold, `N = n_rep * k` unless the `empty_unions` method handling was "drop".

    matrix: ndarray
        A NumPy array of shape `(N,N)` where `N = n_rep * k` for `n_rep` repetitions of k-fold, and
        where `matrix[i,j]` holds the consistency for pairings `i` and `j`

    total_consistency: float
        Given the `N` predictions on the test set, where `N = n_rep * k` for `n_rep` repetitions of
        k-fold, this is the value of the size of the intersection of all error sets divided
        by the size of the union of all those error sets. That is, this is the size of the set of
        all samples that were *always* consistently predicted incorrectly divided by the size of the
        set of all samples that had at least one wrong prediction. When the total_consistency is
        nonzero, this thus means that there are samples which are *always* incorrectly predicted
        regarded of the training set. This thus be thought of as something like a *lower bound* on
        the consistency estimate, where a non-zero value here is indicative / interesting.

    leave_one_out_consistency: float
        Given the `N` predictions on the test set, where `N = n_rep * k` for `n_rep` repetitions of
        k-fold, this is the value of the size of the intersection of all error sets *excluding one*
        divided by the size of the union of all those error sets *excluding the same one*, for each
        excluded error set. See README.md for `p-1` consistency. This is a slightly less punishing
        lower-bound that the total_consistency, and is more symmetric with the pairwise consistency.

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

    def __str__(self) -> str:
        info = []
        info.append(f"Consistencies: {(len(self.consistencies.ravel()))}\n")
        info.append(f"   Mean:  {np.mean(self.consistencies):0.4f}\n")
        info.append(f"   SD:    {np.std(self.consistencies, ddof=1):0.4f}\n")
        info.append(f"   Total: {self.total_consistency:0.4f}\n")
        info.append(f"   LOO:   {self.leave_one_out_consistency:0.4f}\n")
        if self.test_accs is not None:
            accs = self.test_accs.ravel()
            info.append(f"Test Accuracies: {(len(accs))}\n")
            info.append(f"   Mean: {np.mean(accs):0.4f}\n")
            info.append(f"   SD:   {np.std(accs, ddof=1):0.4f}\n")
        if self.fold_accs is not None:
            accs = self.fold_accs.ravel()
            info.append(f"Fold Accuracies: {len(accs)}\n")
            info.append(f"   Mean: {np.mean(accs):0.4f}\n")
            info.append(f"   SD:   {np.std(accs, ddof=1):0.4f}\n")
        return "".join(info)

    def summary(self, name: Optional[str] = None) -> DataFrame:
        df = DataFrame(
            {
                "Mean": [np.mean(self.consistencies.ravel())],
                "SD": [np.std(self.consistencies.ravel(), ddof=1)],
                "Total": [self.total_consistency],
                "LOO": [self.leave_one_out_consistency],
            },
            index=[name] if name is not None else [0],
        )
        if self.test_accs is not None:
            accs = self.test_accs.ravel()
            df["Test-Mean-Acc"] = np.mean(accs)
            df["Test-SD-Acc"] = np.std(accs, ddof=1)
        if self.fold_accs is not None:
            accs = self.fold_accs.ravel()
            df["Fold-Mean-Acc"] = np.mean(accs)
            df["Fold-SD-Acc"] = np.std(accs, ddof=1)
        return df

