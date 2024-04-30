import pytest
import numpy as np

from pyedpyper.models.ridge_regression import RidgeRegression

rng = np.random.default_rng()


def test_error_rr():
    X = rng.random((1000, 5))
    y1 = rng.random((1001, 1))
    y2 = rng.random((1000, 2))
    rr = RidgeRegression("ridge")
    with pytest.raises(ValueError):
        rr.fit(X, y1, 1)
    with pytest.raises(ValueError):
        rr.fit(X, y2, 1)
