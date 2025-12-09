import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)


class SolarLunarTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        for col in self.features:
            minutes = X_copy[col].dt.hour * 60 + X_copy[col].dt.minute

            is_nan = minutes.isna()
            minutes_safe = minutes.fillna(0)

            normalized_time = minutes_safe / 1440.0

            sin_values = np.sin(2 * np.pi * normalized_time)
            cos_values = np.cos(2 * np.pi * normalized_time)

            X_copy[f'{col}_sin'] = np.where(is_nan, np.nan, sin_values)
            X_copy[f'{col}_cos'] = np.where(is_nan, np.nan, cos_values)

        return X_copy[[c for c in X_copy.columns if '_sin' in c or '_cos' in c]]
