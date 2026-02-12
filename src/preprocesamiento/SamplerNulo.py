from imblearn.base import BaseSampler

class SamplerNulo(BaseSampler):
    def _fit_resample(self, X, y):
        return X, y

    def fit(self, X, y=None):
        return self