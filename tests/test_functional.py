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
        assert np.all(matrix.diagonal().ravel() == -1)
        assert np.all(matrix[np.tril_indices(p)].ravel() == -1)
        assert np.all(matrix[np.triu_indices(p, 1)].ravel() >= 0)
