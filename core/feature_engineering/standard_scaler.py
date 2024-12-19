import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, ["mean_", "scale_"])
        X_scaled = (X - self.mean_) / self.scale_
        return X_scaled

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
