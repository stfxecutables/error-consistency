import numpy as np
import pytest
from typing import Any, Dict, List, Optional, Tuple, Union
from typing import cast, no_type_check
from typing_extensions import Literal

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from error_consistency.consistency import ErrorConsistencyKFoldHoldout


class TestConsistency:
    def test_sanity(self, capsys: Any) -> None:
        with capsys.disabled():
            X, y = load_iris(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            knn_args = dict(n_neighbors=10, n_jobs=-1)
            errcon = ErrorConsistencyKFoldHoldout(
                model=KNN, x=X_train, y=y_train, n_splits=5, model_args=knn_args
            )
            results = errcon.evaluate(
                x_test=X_test,
                y_test=y_test,
                repetitions=100,
                save_fold_accs=True,
                save_test_accs=True,
                save_test_errors=True,
                turbo=True,
            )

            print("Mean pairwise consistency ....... ", np.round(np.mean(results.consistencies), 4))
            print(
                "Mean leave-one-out consistency .. ", np.round(results.leave_one_out_consistency, 4)
            )
            print("Total consistency ............... ", np.round(results.total_consistency, 4))
            print("sd (pairwise) ................... ", np.std(results.consistencies, ddof=1))
            print(
                "sd (leave-one-out) .............. ",
                np.std(results.leave_one_out_consistency, ddof=1),
            )

    def test_numba(self, capsys: Any) -> None:
        with capsys.disabled():
            X, y = load_iris(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            knn_args = dict(n_neighbors=10, n_jobs=-1)
            errcon = ErrorConsistencyKFoldHoldout(
                model=KNN, x=X_train, y=y_train, n_splits=5, model_args=knn_args
            )
            for _ in range(10):
                errcon.evaluate(
                    x_test=X_test,
                    y_test=y_test,
                    repetitions=100,
                    save_fold_accs=True,
                    save_test_accs=True,
                    save_test_errors=True,
                    turbo=True,
                )

    def test_slow(self, capsys: Any) -> None:
        with capsys.disabled():
            X, y = load_iris(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            knn_args = dict(n_neighbors=10, n_jobs=-1)
            errcon = ErrorConsistencyKFoldHoldout(
                model=KNN, x=X_train, y=y_train, n_splits=5, model_args=knn_args
            )
            for _ in range(10):
                results = errcon.evaluate(
                    x_test=X_test,
                    y_test=y_test,
                    repetitions=100,
                    save_fold_accs=True,
                    save_test_accs=True,
                    save_test_errors=True,
                    turbo=False,
                )

    def test_parallel_reps(self, capsys: Any) -> None:
        with capsys.disabled():
            X, y = load_iris(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            knn_args = dict(n_neighbors=10, n_jobs=1)
            errcon = ErrorConsistencyKFoldHoldout(
                model=KNN, x=X_train, y=y_train, n_splits=5, model_args=knn_args
            )
            results = errcon.evaluate(
                x_test=X_test,
                y_test=y_test,
                repetitions=1000,
                save_fold_accs=True,
                save_test_accs=True,
                save_test_errors=True,
                parallel_reps=True,
                turbo=True,
            )
            print("Mean pairwise consistency ....... ", np.round(np.mean(results.consistencies), 4))
            print(
                "Mean leave-one-out consistency .. ", np.round(results.leave_one_out_consistency, 4)
            )
            print("Total consistency ............... ", np.round(results.total_consistency, 4))
            print("sd (pairwise) ................... ", np.std(results.consistencies, ddof=1))
            print(
                "sd (leave-one-out) .............. ",
                np.std(results.leave_one_out_consistency, ddof=1),
            )

    def test_parallel_all(self, capsys: Any) -> None:
        with capsys.disabled():
            X, y = load_iris(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            knn_args = dict(n_neighbors=10, n_jobs=1)
            errcon = ErrorConsistencyKFoldHoldout(
                model=KNN, x=X_train, y=y_train, n_splits=5, model_args=knn_args
            )
            results = errcon.evaluate(
                x_test=X_test,
                y_test=y_test,
                repetitions=1000,
                save_fold_accs=True,
                save_test_accs=True,
                save_test_errors=True,
                parallel_reps=True,
                loo_parallel=True,
                turbo=True,
            )
            print("Mean pairwise consistency ....... ", np.round(np.mean(results.consistencies), 4))
            print(
                "Mean leave-one-out consistency .. ", np.round(results.leave_one_out_consistency, 4)
            )
            print("Total consistency ............... ", np.round(results.total_consistency, 4))
            print("sd (pairwise) ................... ", np.std(results.consistencies, ddof=1))
            print(
                "sd (leave-one-out) .............. ",
                np.std(results.leave_one_out_consistency, ddof=1),
            )
