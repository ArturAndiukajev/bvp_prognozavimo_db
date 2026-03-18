import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nowcasting.models.base import BaseNowcastModel

logger = logging.getLogger("nowcast_ml")

class ElasticNetNowcast(BaseNowcastModel):
    """
    ElasticNet regression handling multi-series panels.
    Automatically scales inputs and tunes L1/L2 ratios via cross-validation.

    Parameters
    ----------
    fill_strategy : str
        How to fill NaN values before fitting. Options:
        'zero' (default), 'mean', 'median', 'ffill_then_zero'.
    cv : int
        Number of cross-validation folds for ElasticNetCV (default 5).
    max_iter : int
        Maximum iterations for coordinate descent (default 2000).
    l1_ratio : float or list of float
        L1 ratio(s) to search over in ElasticNetCV (default [0.1,0.5,0.7,0.9,0.95,1.0]).
    """
    def __init__(
        self,
        target_col: str,
        horizon: int = 1,
        fill_strategy: str = "zero",
        cv: int = 5,
        max_iter: int = 2000,
        l1_ratio: Union[float, List[float], None] = None,
        **kwargs,
    ):
        super().__init__(target_col, horizon, **kwargs)
        seed = kwargs.get("random_state", kwargs.get("seed", 42))
        self.fill_strategy = fill_strategy
        _l1 = l1_ratio if l1_ratio is not None else [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]
        
        cv_splitter = TimeSeriesSplit(n_splits=cv)
        
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNetCV(cv=cv_splitter, random_state=seed, max_iter=max_iter, l1_ratio=_l1))
        ])

    # ------------------------------------------------------------------
    def _fill(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.fill_strategy == "mean":
            return df.fillna(df.mean()).fillna(0)
        elif self.fill_strategy == "median":
            return df.fillna(df.median()).fillna(0)
        elif self.fill_strategy == "ffill_then_zero":
            return df.ffill().fillna(0)
        else:  # 'zero' (default)
            return df.fillna(0)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        valid_idx = y_train.notna()
        X_train_clean = self._fill(X_train.loc[valid_idx])
        y_train_clean = y_train.loc[valid_idx]
        self.model.fit(X_train_clean, y_train_clean)
        self.is_fitted = True
        return self

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        preds = self.model.predict(self._fill(X_test))
        return pd.Series(preds, index=X_test.index, name=f"{self.target_col}_pred")

class LightGBMNowcast(BaseNowcastModel):
    """
    Gradient Boosting model for capturing non-linear macroeconomic dynamics.
    Naturally handles NaNs in X_train (ragged edges).

    All hyperparameter kwargs are forwarded directly to LGBMRegressor.
    Defaults deliberately match the original baseline.
    """
    def __init__(
        self,
        target_col: str,
        horizon: int = 1,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        **kwargs,
    ):
        super().__init__(target_col, horizon, **kwargs)
        seed = kwargs.get("random_state", kwargs.get("seed", 42))
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=seed,
            n_jobs=1,   # 1 per worker to avoid nested parallelism in grid search
            verbose=-1,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        valid_idx = y_train.notna()
        y_train_clean = y_train.loc[valid_idx]
        # LightGBM natively handles NaNs — leave X intact for ragged-edge benefits.
        X_train_clean = X_train.loc[valid_idx]
        self.model.fit(X_train_clean, y_train_clean)
        self.is_fitted = True
        return self

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        preds = self.model.predict(X_test)
        return pd.Series(preds, index=X_test.index, name=f"{self.target_col}_pred")
