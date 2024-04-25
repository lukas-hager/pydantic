import pandas as pd
import numpy as np

class LinearRegression:
    def __init__(self, name):
        self.name = name
    def _check_dims(self):
        """Check input dimensions"""
        return self.X.shape[0] == self.y.shape[0] and self.y.shape[1] == 1
    def fit(self, X: np.array, y: np.array):
        """Fit a linear regression"""
        self.X = X
        self.y = y
        if not self._check_dims():
            raise ValueError('Dimensions are not correct')
        self.beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        self.resid = (self.y-self.X@self.beta).flatten()
        self.cov = np.linalg.inv(self.X.T@self.X) * np.inner(self.resid, self.resid)
    def summary(self) -> pd.DataFrame:
        """Produce a regression table"""
        ses = np.sqrt(np.diag(self.cov))
        data = {
            'coef': self.beta.flatten(),
            'se': ses,
            't-stat': self.beta.flatten() / ses
        }
        return pd.DataFrame(data)