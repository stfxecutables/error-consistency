import numpy as np
from error_consistency.functional import get_y_error, error_consistencies
import pytest

from tests.testutils import random_preds


@pytest.mark.fast
def test_get_y_error() -> None:
    # test throws errors

    # test random arrays
    for _ in range(1000):
        y_pred, y_true = random_preds(1000, 100, n_preds=1, dims=1)
        get_y_error(y_pred, y_true, sample_dim=1)

    for _ in range(1000):
        y_pred, y_true = random_preds(1000, 100, n_preds=1, dims=2)
        get_y_error(y_pred, y_true, sample_dim=0)


@pytest.mark.fast
def test_get_consistencies() -> None:
    for p in range(2, 100):
        y_preds, y_true = random_preds(1000, 100, n_preds=p, dims=1)
        cs, matrix = error_consistencies(y_preds, y_true, sample_dim=0)
        assert len(cs) == p * (p - 1) / 2
        assert matrix.shape == (p, p)
        assert np.all(matrix.diagonal().ravel() == 1)
        assert np.all(matrix[np.tril_indices(p, 1)].ravel() >= 0)
        assert np.all(matrix[np.triu_indices(p, 1)].ravel() >= 0)


def test_trivial_consistencies() -> None:
    # Perfect consistency
    y_preds = [np.zeros(10) for _ in range(10)]
    y_true = np.ones(10)
    cs, matrix = error_consistencies(y_preds, y_true)
    assert np.all(cs == 1)
    assert np.all(matrix.ravel() == 1)

    # also perfect consistency
    y_preds = [np.ones(10) for _ in range(10)]
    y_true = np.ones(10)
    # testing with empty_unions="nan"
    cs, matrix = error_consistencies(y_preds, y_true)
    assert np.all(np.isnan(cs))
    assert np.all(matrix.diagonal().ravel() == 1)
    matrix[matrix == 1] = np.nan
    assert np.all(np.isnan(matrix))

    # testing with empty_unions="drop"
    cs, matrix = error_consistencies(y_preds, y_true, empty_unions="drop")
    assert len(cs) == 0
    assert np.all(matrix.diagonal().ravel() == 1)
    matrix[matrix == 1] = np.nan
    assert np.all(np.isnan(matrix))


def test_nontrivial_consistencies() -> None:
    # constructed non-trivial error consistency of zero
    for length in range(10, 100):
        y_preds = [np.ones(length) for _ in range(length)]
        y_true = np.ones(length)
        for i, y_pred in enumerate(y_preds):
            # just make one prediction wrong each time, but in a different location
            # this is zero error consistency (all intersections are empty)
            y_pred[i] = 0
        cs, matrix = error_consistencies(y_preds, y_true)
        assert np.all(cs == 0)
        assert np.all(matrix.diagonal().ravel() == 1)
        matrix[matrix == 1] = 0
        assert np.all(matrix == 0)

    # constructed 33.3% error consistency
    for length in range(10, 100):
        y_preds = [np.ones(length) for _ in range(length + 1)]
        y_true = np.ones(length)
        for i in range(length):
            # prediction on first sample (sample 0) is always wrong
            # one different sample is predicted wrong each time
            # when comparing, the intersection is thus always sample 0, so size of intersection is 1
            # The union or errors for predictions i, j is thus {0, i, j} = 3
            # thus our error is 0.33333
            y_preds[i][0] = 0
            y_preds[i][i] = 0
        y_preds = y_preds[1:-1]  # first and last predictions have only one error
        cs, matrix = error_consistencies(y_preds, y_true)
        assert np.allclose(cs, np.full(cs.shape, 1 / 3))
        assert np.all(matrix.diagonal().ravel() == 1)
        matrix[matrix == 1] = 1.0 / 3
        assert np.allclose(matrix, np.full(matrix.shape, 1 / 3))
