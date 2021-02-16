import pytest
from tests.loading import load_diabetes, load_park, load_trans, load_SPECT


@pytest.mark.fast
def test_loading_functions() -> None:
    x, y = load_diabetes()
    assert x.shape == (768, 8)
    assert y.shape == (768,)

    x, y = load_park()
    assert x.shape == (195, 23)
    assert y.shape == (195,)

    x, y = load_trans()
    assert x.shape == (748, 4)
    assert y.shape == (748,)

    x, y = load_SPECT()
    assert x.shape == (267, 22)
    assert y.shape == (267,)
