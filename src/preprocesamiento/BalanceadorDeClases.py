import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class BalanceadorDeClases(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.media_ = None
        self.std_ = None

    def fit(self, X, y=None):
        # Calcular media y desviación
        X = np.array(X)
        self.media_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=0)
        # Evitar dividir por cero
        self.std_[self.std_ == 0] = 1
        return self

    def transform(self, X):
        X = np.array(X)
        return (X - self.media_) / self.std_
