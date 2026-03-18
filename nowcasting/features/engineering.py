import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger("nowcast_engineering")

class TimeSeriesFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Generates standard macro-financial lags and rolling features.
    Should be fit on training data and subsequently used to transform any window.
    """
    def __init__(self, lags: list[int] = [1, 3, 6], rolling_windows: list[int] = [3, 6]):
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.feature_names_in_ = None

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = X.columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Generating features for {X.shape[1]} series over {X.shape[0]} periods...")
        features = [X.copy()]
        
        # Lags
        for lag in self.lags:
            shifted = X.shift(lag)
            shifted.columns = [f"{col}_lag{lag}" for col in X.columns]
            features.append(shifted)

        # Rolling Means
        for w in self.rolling_windows:
            rolled = X.rolling(window=w, min_periods=1).mean()
            rolled.columns = [f"{col}_roll_mean_{w}" for col in X.columns]
            features.append(rolled)

        # Rolling Volatility (std)
        for w in self.rolling_windows:
            rolled = X.rolling(window=w, min_periods=2).std()
            rolled.columns = [f"{col}_roll_std_{w}" for col in X.columns]
            features.append(rolled)

        X_out = pd.concat(features, axis=1)
        return X_out
