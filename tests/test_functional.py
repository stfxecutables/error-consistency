import numpy as np
import pytest

from error_consistency.functional import error_consistencies, get_y_error

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
        cs, matrix, unpredictable, predictable = error_consistencies(y_preds, y_true, sample_dim=0)
        assert len(cs) == p * (p - 1) / 2
        assert matrix.shape == (p, p)
        assert np.all(matrix.diagonal().ravel() == 1)
        assert np.all(matrix[np.tril_indices(p, 1)].ravel() >= 0)
        assert np.all(matrix[np.triu_indices(p, 1)].ravel() >= 0)


def test_trivial_consistencies() -> None:
    # Perfect consistency
    y_preds = [np.zeros(10) for _ in range(10)]
    y_true = np.ones(10)
    cs, matrix, unpredictable, predictable = error_consistencies(y_preds, y_true)
    assert np.all(cs == 1)
    assert np.all(matrix.ravel() == 1)

    # also perfect consistency
    y_preds = [np.ones(10) for _ in range(10)]
    y_true = np.ones(10)
    # testing with empty_unions="nan"
    cs, matrix, unpredictable, predictable = error_consistencies(y_preds, y_true)
    assert np.all(np.isnan(cs))
    assert np.all(matrix.diagonal().ravel() == 1)
    matrix[matrix == 1] = np.nan
    assert np.all(np.isnan(matrix))

    # testing with empty_unions="drop"
    cs, matrix, unpredictable, predictable = error_consistencies(
        y_preds, y_true, empty_unions="drop"
    )
    assert len(cs) == 0
    assert np.all(matrix.diagonal().ravel() == 1)
    matrix[matrix == 1] = np.nan
    assert np.all(np.isnan(matrix))


def test_nontrivial_consistencies() -> None:
    """The results below are a little bit surprising. Note a mean error consistency of 0.333 means
    that on average, on any pairing of predictions, there are three errors, and 1 of those three
    errors are in the same location / on the same sample, i.e 33.3% of the errors *in a pairing*
    occur in the same location.

    However, this may be unintuitive. In the constructed example below, every classifier makes
    only two errors: one on the first value, and one on another value. Which is to say, 50% of the
    errors are consistently made on the same sample, or that any two classifiers in the example
    below will in fact agree on all predictions expect for one sample.

    The key is to note that the error consistency is about *pairings*. Inthe example below, we have
    models M_i and M_j, and for any i, j, we have the predictions of sample indices

        M_i(0) == False         M_j(0) == False
        M_i(i) == False         M_j(i) == True
        M_i(j) == True          M_j(j) == False

    That is, there are in fact *3* values in any given comparison that have at least one error. Now,
    by construction, we *know* that in fact no classifier ever makes more than 2 errors. In effect,
    we *want* to exclude sample 0 from consideration, because we know all classifiers make a mistake
    on this value (are consistent on this value).

    We might want two more measures: the size of the set which *always* is in error (intersection of
    all error sets) and the max-set (union of all error sets). Without these two the error
    consistency measure is somewhat difficult to interpret for a given set of predictions.
    """
    # constructed non-trivial error consistency of zero
    for length in range(10, 100):
        y_preds = [np.ones(length) for _ in range(length)]
        y_true = np.ones(length)
        for i, y_pred in enumerate(y_preds):
            # just make one prediction wrong each time, but in a different location
            # this is zero error consistency (all intersections are empty)
            y_pred[i] = 0
        cs, matrix, unpredictable, predictable = error_consistencies(y_preds, y_true)
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
        cs, matrix, unpredictable, predictable = error_consistencies(y_preds, y_true)
        assert np.allclose(cs, np.full(cs.shape, 1 / 3))
        assert np.all(matrix.diagonal().ravel() == 1)
        matrix[matrix == 1] = 1.0 / 3
        assert np.allclose(matrix, np.full(matrix.shape, 1 / 3))
