import pytest
import numpy as np

from pyedpyper.models.linear_regression import LinearRegression
from pyedpyper.models.ridge_regression import RidgeRegression

rng = np.random.default_rng()


def test_fit_no_error():
    X = rng.random((1000, 5))
    beta = np.arange(1.0, 6.0).reshape(-1, 1)
    y = X @ beta
    lr = LinearRegression("ols")
    lr.fit(X, y)
    assert np.allclose(lr.beta, beta, rtol=0.001, atol=0)


def test_comparison():
    X = rng.random((1000, 5))
    beta = np.arange(1.0, 6.0).reshape(-1, 1)
    y = X @ beta
    lr = LinearRegression("ols")
    lr.fit(X, y)
    rr = RidgeRegression("ridge")
    rr.fit(X, y, 0)
    assert np.allclose(lr.beta, rr.beta, rtol=0, atol=1e-6)


def test_error_lr():
    X = rng.random((1000, 5))
    y1 = rng.random((1001, 1))
    y2 = rng.random((1000, 2))
    lr = LinearRegression("ols")
    with pytest.raises(ValueError):
        lr.fit(X, y1)
    with pytest.raises(ValueError):
        lr.fit(X, y2)
