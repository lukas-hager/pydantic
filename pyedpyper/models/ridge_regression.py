import pandas as pd
import numpy as np

from .. import utils as ut

class RidgeRegression:
    def __init__(self, name):
        self.name = name
    def _check_dims(self):
        """Check input dimensions"""
        return self.X.shape[0] == self.y.shape[0] and self.y.shape[1] == 1\
             and isinstance(self.alpha, (float,int))
    def fit(self, X: np.array, y: np.array, alpha: float):
        """Fit a linear regression"""
        self.X = X
        self.y = y
        self.alpha = alpha
        if not self._check_dims():
            raise ValueError('Dimensions are not correct')
        denom = np.linalg.inv(self.X.T @ self.X + self.alpha * np.eye(self.X.shape[1]))
        num = self.X.T @ self.y
        self.beta = denom @ num
    def summary(self) -> pd.DataFrame:
        """Produce a regression table"""
        
        return ut.summary(beta = self.beta)