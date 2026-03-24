"""
bridge_equation.py — Bridge Equation Nowcast Model
===================================================

A two-stage mixed-frequency nowcasting approach:

Stage 1 (Bridge):
    For each high-frequency predictor, fit an AR(p) model on the training
    history and forecast it forward to fill the gap to the end of the current
    low-frequency period (e.g. complete a quarter using available monthly data).

Stage 2 (Regression):
    Aggregate the (real + forecasted) high-frequency predictors to the low
    frequency, then run an ElasticNetCV regression of the target on them.

References
----------
Baffigi, A., Golinelli, R., & Parigi, G. (2004). Bridge models to forecast
the euro area GDP. *International Journal of Forecasting*, 20(3), 447-460.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from nowcasting.models.base import BaseNowcastModel

logger = logging.getLogger("nowcast_bridge")


# ---------------------------------------------------------------------------
# AR bridge — forecast one high-frequency series to a target date
# ---------------------------------------------------------------------------

def _ar_forecast(series: pd.Series, target_date: pd.Timestamp, ar_lags: int) -> pd.Series:
    """
    Fit AR(ar_lags) on `series` and forecast forward until `target_date`.
    Returns a pd.Series that is the concatenation of known values + forecasts.
    """
    s = series.dropna()
    if len(s) < ar_lags + 2:
        # Not enough history — just forward-fill
        return series.ffill()

    # Build AR feature matrix
    y = s.values
    X = np.column_stack([y[i : len(y) - ar_lags + i] for i in range(ar_lags)])
    y_target = y[ar_lags:]

    if len(y_target) < 2:
        return series.ffill()

    model = LinearRegression().fit(X, y_target)

    # Determine how many steps ahead we need
    last_known = s.index[-1]
    if target_date <= last_known:
        return series

    # Infer frequency from series index
    if len(series.index) >= 2:
        freq_delta = (series.index[-1] - series.index[-2])
    else:
        freq_delta = pd.Timedelta(days=30)

    steps = 0
    current_date = last_known
    future_dates = []
    while current_date < target_date:
        current_date = current_date + freq_delta
        future_dates.append(current_date)
        steps += 1
        if steps > 36:  # safety cap
            break

    if not future_dates:
        return series

    # Iterative multi-step forecast
    history = list(y[-ar_lags:])
    forecasts = []
    for _ in range(len(future_dates)):
        x_new = np.array(history[-ar_lags:]).reshape(1, -1)
        next_val = float(model.predict(x_new)[0])
        forecasts.append(next_val)
        history.append(next_val)

    forecast_s = pd.Series(forecasts, index=future_dates)
    return pd.concat([series, forecast_s])


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def _aggregate_to_lf(hf_series: pd.Series, lf_freq: str, rule: str = "mean") -> pd.Series:
    """Aggregate a high-frequency series to a low-frequency index."""
    if rule == "last":
        return hf_series.resample(lf_freq).last()
    elif rule == "sum":
        return hf_series.resample(lf_freq).sum(min_count=1)
    else:
        return hf_series.resample(lf_freq).mean()


# ---------------------------------------------------------------------------
# Bridge Equation Model
# ---------------------------------------------------------------------------

class BridgeEquationNowcast(BaseNowcastModel):
    """
    Bridge Equation Nowcasting model.

    Parameters
    ----------
    target_col : str
        Label for the target series.
    horizon : int
        Low-frequency periods ahead to forecast.
    freq_ratio : int | None
        High-freq / low-freq ratio (e.g. 3 for monthly→quarterly).
        Auto-inferred if None.
    ar_lags : int
        Number of lags for the AR bridge model used to complete quarters.
    lf_freq : str
        pandas resample alias for the low-frequency target (e.g. 'QE').
    agg_rule : str
        Aggregation rule when collapsing HF→LF ('mean', 'last', 'sum').
    regression_model : str
        One of 'linear', 'ridge', 'lasso', or 'elasticnet'.
    regression_kwargs : dict
        Optional kwargs to pass to the chosen regression model (e.g. l1_ratio, cv, alphas).
    """

    def __init__(
        self,
        target_col: str,
        horizon: int = 1,
        freq_ratio: Optional[int] = None,
        ar_lags: int = 3,
        lf_freq: str = "QE",
        agg_rule: str = "mean",
        regression_model: str = "elasticnet",
        regression_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(target_col, horizon, **kwargs)
        self.freq_ratio = freq_ratio
        self.ar_lags = ar_lags
        self.lf_freq = lf_freq
        self.agg_rule = agg_rule
        self.regression_model_type = regression_model.lower()
        self.regression_kwargs = regression_kwargs or {}
        self.scaler = StandardScaler()
        
        # Pull seed from kwargs (passed from experiment runner or main.py)
        seed = kwargs.get("random_state", kwargs.get("seed", 123))
        
        # Build regression model
        if self.regression_model_type == "linear":
            self.reg_model = LinearRegression(**self.regression_kwargs)
        elif self.regression_model_type == "ridge":
            cv_val = self.regression_kwargs.pop("cv", 5)
            cv = TimeSeriesSplit(n_splits=cv_val) if isinstance(cv_val, int) else cv_val
            self.reg_model = RidgeCV(cv=cv, **self.regression_kwargs)
        elif self.regression_model_type == "lasso":
            cv_val = self.regression_kwargs.pop("cv", 5)
            cv = TimeSeriesSplit(n_splits=cv_val) if isinstance(cv_val, int) else cv_val
            max_iter = self.regression_kwargs.pop("max_iter", 3000)
            self.reg_model = LassoCV(cv=cv, max_iter=max_iter, random_state=seed, **self.regression_kwargs)
        else: # default to elasticnet
            cv_val = self.regression_kwargs.pop("cv", 5)
            cv = TimeSeriesSplit(n_splits=cv_val) if isinstance(cv_val, int) else cv_val
            max_iter = self.regression_kwargs.pop("max_iter", 3000)
            self.reg_model = ElasticNetCV(cv=cv, random_state=seed, max_iter=max_iter, **self.regression_kwargs)
            
        self._hf_series_train: Optional[pd.DataFrame] = None
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    def _bridge_and_aggregate(
        self,
        X_hf: pd.DataFrame,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        For each HF series, bridge forward to end_date, then aggregate to LF.
        """
        lf_panels = {}
        for col in X_hf.columns:
            bridged = _ar_forecast(X_hf[col], target_date=end_date, ar_lags=self.ar_lags)
            lf_panels[col] = _aggregate_to_lf(bridged, self.lf_freq, self.agg_rule)

        return pd.DataFrame(lf_panels)

    # ------------------------------------------------------------------
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BridgeEquationNowcast":
        """
        X_train : high-frequency panel (e.g. monthly)
        y_train : low-frequency target (e.g. quarterly)
        """
        if X_train.empty or y_train.dropna().empty:
            logger.warning("BridgeEquationNowcast.fit: empty train data, skipping.")
            return self

        self._hf_series_train = X_train.copy()
        end_date = X_train.index.max()

        logger.info(
            f"BridgeEquationNowcast.fit: X={X_train.shape}, "
            f"y_obs={y_train.dropna().shape[0]}, "
            f"ar_lags={self.ar_lags}, lf_freq={self.lf_freq}"
        )

        # Stage 1: Bridge + aggregate
        X_lf = self._bridge_and_aggregate(X_train, end_date)

        # Stage 2: Align with target
        common_idx = X_lf.index.intersection(y_train.dropna().index)
        if len(common_idx) < 3:
            logger.warning(
                f"BridgeEquationNowcast.fit: only {len(common_idx)} aligned observations — "
                "Bridge may not generalise well."
            )
            if len(common_idx) == 0:
                return self

        X_aligned = X_lf.loc[common_idx].fillna(0)
        y_aligned = y_train.loc[common_idx]

        self._feature_cols = X_aligned.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_aligned)
        self.reg_model.fit(X_scaled, y_aligned.values)

        self.is_fitted = True
        logger.info(f"BridgeEquationNowcast fitted on {len(y_aligned)} low-freq observations.")
        return self

    # ------------------------------------------------------------------
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        X_test : high-frequency panel (may be a partial period / ragged right edge).
        Returns a Series at low-frequency dates.
        """
        if not self.is_fitted:
            raise ValueError("BridgeEquationNowcast not fitted yet.")

        end_date = X_test.index.max()
        # Combine train history (for AR context) + test data
        X_combined = pd.concat([self._hf_series_train, X_test]).drop_duplicates()

        # Bridge and aggregate
        X_lf = self._bridge_and_aggregate(X_combined, end_date)

        # Keep only dates within X_test range
        try:
            lf_offset = pd.tseries.frequencies.to_offset(self.lf_freq)
        except Exception:
            lf_offset = pd.offsets.MonthEnd(3)
            
        lf_in_test = X_lf.loc[
            (X_lf.index >= X_test.index.min()) & (X_lf.index <= end_date + lf_offset)
        ]

        if lf_in_test.empty:
            lf_in_test = X_lf.tail(1)

        # Align feature columns
        for col in self._feature_cols:
            if col not in lf_in_test.columns:
                lf_in_test[col] = 0.0
        X_pred = lf_in_test[self._feature_cols].fillna(0)

        X_scaled = self.scaler.transform(X_pred)
        preds = self.reg_model.predict(X_scaled)

        return pd.Series(preds, index=lf_in_test.index, name=f"{self.target_col}_pred")
