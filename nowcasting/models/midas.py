"""
midas.py — U-MIDAS (Unrestricted MIDAS) Nowcast Model
=======================================================

Mixed-Data Sampling regression where high-frequency predictors are stacked
into a flat feature vector and regressed on a low-frequency target.

MIDAS intuition
---------------
Given a quarterly target y_t and monthly predictors x_{t,m} (m=1,2,3 within quarter t):
  y_t = α + β₁·x_{t,1} + β₂·x_{t,2} + β₃·x_{t,3} + ε_t

The *unrestricted* version (U-MIDAS) places no polynomial constraint on the
β weights. This is adequate when freq_ratio ≤ 6 (e.g. M→Q=3, W→M≈4).

Interface
---------
X_train : pd.DataFrame with DatetimeIndex at HIGH frequency (e.g. monthly)
y_train : pd.Series with DatetimeIndex at LOW frequency (e.g. quarterly)

The model aligns internally and handles ragged right edges (the current
high-frequency period may be incomplete).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from nowcasting.models.base import BaseNowcastModel

logger = logging.getLogger("nowcast_midas")


def _infer_freq_ratio(hf_index: pd.DatetimeIndex, lf_index: pd.DatetimeIndex) -> int:
    """
    Heuristically infer how many high-freq periods map to one low-freq period.
    E.g. monthly X, quarterly y → ratio = 3.
    """
    if len(hf_index) < 2 or len(lf_index) < 2:
        return 3  # default
    hf_delta = (hf_index[-1] - hf_index[0]).days / max(len(hf_index) - 1, 1)
    lf_delta = (lf_index[-1] - lf_index[0]).days / max(len(lf_index) - 1, 1)
    ratio = max(1, round(lf_delta / hf_delta))
    return ratio


def _build_midas_features(
    X_hf: pd.DataFrame,
    y_lf: pd.Series,
    freq_ratio: int,
    n_lags: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    For each low-frequency observation date, stack the `freq_ratio * n_lags`
    most recent high-frequency values into a flat row.

    Returns (X_aligned, y_aligned) with the low-frequency index.
    """
    rows = []
    y_values = []
    lf_dates = []

    # Align using the low-freq target dates as the anchor
    for lf_date, lf_val in y_lf.items():
        if pd.isna(lf_val):
            continue

        # Find the high-freq observations up to and including lf_date
        hf_slice = X_hf.loc[X_hf.index <= lf_date]
        n_needed = freq_ratio * n_lags

        if len(hf_slice) < freq_ratio:  # need at least one full period
            continue

        # Take up to n_needed most-recent rows; pad with NaN if not enough history
        if len(hf_slice) >= n_needed:
            window = hf_slice.iloc[-n_needed:]
        else:
            pad_len = n_needed - len(hf_slice)
            pad = pd.DataFrame(
                np.nan, index=range(pad_len), columns=X_hf.columns
            )
            window = pd.concat([pad, hf_slice])

        # Flatten: cols ordered as [lag_1_var1, lag_1_var2, ..., lag_k_varn]
        row_vals = window.values.flatten(order="C")
        rows.append(row_vals)
        y_values.append(lf_val)
        lf_dates.append(lf_date)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=float)

    n_cols = freq_ratio * n_lags * X_hf.shape[1]
    col_names = [
        f"{col}_lag{lag}_t{t}"
        for lag in range(n_lags, 0, -1)
        for t in range(freq_ratio, 0, -1)
        for col in X_hf.columns
    ]

    X_out = pd.DataFrame(rows, index=lf_dates, columns=col_names[:n_cols])
    y_out = pd.Series(y_values, index=lf_dates, name=y_lf.name)
    return X_out, y_out


def _build_midas_features_for_predict(
    X_hf: pd.DataFrame,
    predict_lf_dates: pd.DatetimeIndex,
    freq_ratio: int,
    n_lags: int,
) -> pd.DataFrame:
    """Like _build_midas_features but without requiring a target value."""
    rows = []
    lf_dates = []

    for lf_date in predict_lf_dates:
        hf_slice = X_hf.loc[X_hf.index <= lf_date]
        n_needed = freq_ratio * n_lags

        if len(hf_slice) == 0:
            rows.append(np.full(n_needed * X_hf.shape[1], np.nan))
            lf_dates.append(lf_date)
            continue

        if len(hf_slice) >= n_needed:
            window = hf_slice.iloc[-n_needed:]
        else:
            pad_len = n_needed - len(hf_slice)
            pad = pd.DataFrame(np.nan, index=range(pad_len), columns=X_hf.columns)
            window = pd.concat([pad, hf_slice])

        rows.append(window.values.flatten(order="C"))
        lf_dates.append(lf_date)

    if not rows:
        return pd.DataFrame()

    n_cols = freq_ratio * n_lags * X_hf.shape[1]
    col_names = [
        f"{col}_lag{lag}_t{t}"
        for lag in range(n_lags, 0, -1)
        for t in range(freq_ratio, 0, -1)
        for col in X_hf.columns
    ]
    return pd.DataFrame(rows, index=lf_dates, columns=col_names[:n_cols])


class MIDASNowcast(BaseNowcastModel):
    """
    U-MIDAS Nowcasting model.

    Parameters
    ----------
    target_col : str
        Name/label of the target series.
    horizon : int
        How many low-frequency periods ahead to forecast (default 1).
    freq_ratio : int | None
        High-freq / low-freq ratio (e.g. 3 for monthly→quarterly).
        If None, inferred automatically from the index during fit().
    n_lags : int
        Number of low-frequency lags of high-frequency data to include.
        E.g. n_lags=2 with freq_ratio=3 → 6 monthly observations per predictor.
    """

    def __init__(
        self,
        target_col: str,
        horizon: int = 1,
        freq_ratio: Optional[int] = None,
        n_lags: int = 2,
        lf_freq: str = "QE",
        regression_model: str = "elasticnet",
        regression_kwargs: Optional[dict] = None,
        fill_strategy: str = "zero",
        **kwargs,
    ):
        super().__init__(target_col, horizon, **kwargs)
        self.freq_ratio = freq_ratio
        self.n_lags = n_lags
        self.lf_freq = lf_freq
        self.fill_strategy = fill_strategy
        self.scaler = StandardScaler()
        self._fitted_freq_ratio: int = freq_ratio or 3
        self._feature_cols: list[str] = []

        seed = kwargs.get("seed", kwargs.get("random_state", 42))
        reg_kw = regression_kwargs or {}
        reg_type = regression_model.lower()

        if reg_type == "linear":
            self.model = LinearRegression(**reg_kw)
        elif reg_type == "ridge":
            cv_val = reg_kw.pop("cv", 5)
            cv = TimeSeriesSplit(n_splits=cv_val) if isinstance(cv_val, int) else cv_val
            self.model = RidgeCV(cv=cv, **reg_kw)
        elif reg_type == "lasso":
            cv_val = reg_kw.pop("cv", 5)
            cv = TimeSeriesSplit(n_splits=cv_val) if isinstance(cv_val, int) else cv_val
            max_iter = reg_kw.pop("max_iter", 3000)
            self.model = LassoCV(cv=cv, max_iter=max_iter, random_state=seed, **reg_kw)
        else:  # default elasticnet
            cv_val = reg_kw.pop("cv", 5)
            cv = TimeSeriesSplit(n_splits=cv_val) if isinstance(cv_val, int) else cv_val
            max_iter = reg_kw.pop("max_iter", 3000)
            self.model = ElasticNetCV(cv=cv, random_state=seed, max_iter=max_iter, **reg_kw)

    # ------------------------------------------------------------------
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "MIDASNowcast":
        """
        X_train : high-frequency panel (e.g. monthly)
        y_train : low-frequency target (e.g. quarterly)
        """
        if X_train.empty or y_train.dropna().empty:
            logger.warning("MIDASNowcast.fit: empty train data, skipping.")
            return self

        # Infer freq_ratio if not provided
        if self.freq_ratio is None:
            self._fitted_freq_ratio = _infer_freq_ratio(
                pd.DatetimeIndex(X_train.index), pd.DatetimeIndex(y_train.dropna().index)
            )
        else:
            self._fitted_freq_ratio = self.freq_ratio

        logger.info(
            f"MIDASNowcast.fit: freq_ratio={self._fitted_freq_ratio}, "
            f"n_lags={self.n_lags}, X={X_train.shape}, "
            f"y_obs={y_train.dropna().shape[0]}"
        )

        X_feat, y_aligned = _build_midas_features(
            X_train, y_train.dropna(), self._fitted_freq_ratio, self.n_lags
        )

        if X_feat.empty:
            logger.warning("MIDASNowcast.fit: no aligned samples after feature building.")
            return self

        self._feature_cols = X_feat.columns.tolist()

        # Apply fill strategy
        X_filled = self._fill(X_feat)
        X_scaled = self.scaler.fit_transform(X_filled)
        self.model.fit(X_scaled, y_aligned.values)

        self.is_fitted = True
        logger.info(f"MIDASNowcast fitted on {len(y_aligned)} low-freq observations.")
        return self

    # ------------------------------------------------------------------
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        X_test : high-frequency panel covering the prediction window.
        Returns a Series at low-frequency (quarterly/target) dates.
        """
        if not self.is_fitted:
            raise ValueError("MIDASNowcast not fitted yet.")

        # Derive low-freq prediction dates from X_test using configured lf_freq
        extra_months = 3 if "Q" in self.lf_freq else (1 if "M" in self.lf_freq else 12)
        lf_dates = pd.date_range(
            start=X_test.index.min(),
            end=X_test.index.max() + pd.offsets.MonthEnd(extra_months),
            freq=self.lf_freq,
        )
        lf_dates = lf_dates[lf_dates >= X_test.index.min()]

        if lf_dates.empty:
            lf_dates = pd.DatetimeIndex([X_test.index.max()])

        X_feat = _build_midas_features_for_predict(
            X_test, lf_dates, self._fitted_freq_ratio, self.n_lags
        )

        if X_feat.empty:
            return pd.Series(dtype=float, name=f"{self.target_col}_pred")

        # Align columns (some may be missing for very short X_test)
        for col in self._feature_cols:
            if col not in X_feat.columns:
                X_feat[col] = 0.0
        X_feat = self._fill(X_feat[self._feature_cols])

        X_scaled = self.scaler.transform(X_feat)
        preds = self.model.predict(X_scaled)

        return pd.Series(preds, index=lf_dates[: len(preds)], name=f"{self.target_col}_pred")

    # ------------------------------------------------------------------
    def _fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured fill_strategy to handle NaN values."""
        if self.fill_strategy == "ffill_then_zero":
            return df.ffill().fillna(0)
        elif self.fill_strategy == "mean":
            means = df.mean()
            return df.fillna(means).fillna(0)  # fallback to 0 if all-NaN column
        else:  # default: zero
            return df.fillna(0)
