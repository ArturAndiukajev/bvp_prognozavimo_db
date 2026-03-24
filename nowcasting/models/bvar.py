"""
bvar.py - Bayesian Vector Autoregression with Minnesota Prior
=============================================================

Provides two classes:

    VARNowcast  – baseline frequentist VAR (statsmodels).
    BVARNowcast – true Bayesian VAR with Minnesota shrinkage prior.

Minnesota Prior Overview
------------------------
For coefficients β_{ij,l}  (equation i, variable j, lag l):

    Var(β_{ij,l}) = (λ1 / l^λ3)²          if i == j  (own-lag)
                  = (λ1·λ2 / l^λ3)²         if i != j  (cross-variable)

The prior mean follows the random-walk convention:
    β_{ii,1} = 1   (own first lag)
    all others = 0

Posterior coefficients (analytical, equation-by-equation OLS with Tikhonov):
    β_post = (X'X + V0⁻¹)⁻¹ (X'Y + V0⁻¹ β0)

This is computed once per equation and stored for rolling-window prediction.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from nowcasting.models.base import BaseNowcastModel

logger = logging.getLogger("nowcast_bvar")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_lag_matrix(data: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, Y) regressor matrices from a (T, K) endogenous array.

    Returns
    -------
    X : (T-lags, K*lags + 1)   — stacked lag columns + intercept
    Y : (T-lags, K)             — contemporaneous values
    """
    T, K = data.shape
    rows = T - lags

    X_parts = [np.ones((rows, 1))]          # intercept
    for lag in range(1, lags + 1):
        X_parts.append(data[lags - lag: T - lag, :])

    X = np.hstack(X_parts)       # (rows, 1 + K*lags)
    Y = data[lags:, :]           # (rows, K)
    return X, Y


def _minnesota_prior(K: int, lags: int, lambda1: float, lambda2: float, lambda3: float,
                     sigma: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Minnesota prior mean (beta0) and diagonal precision (V0_inv).

    Parameters
    ----------
    K       : number of variables
    lags    : number of lags
    sigma   : (K,) per-variable residual std (used for scaling cross-variable shrinkage).
              If None, uses 1 (no scaling).

    Returns
    -------
    beta0   : (1 + K*lags, K)  — prior means (RW convention)
    V0_diag : (1 + K*lags, K)  — diagonal of prior covariance V0
              V0_inv is built as 1/V0_diag inside the solver.
    """
    n_coef = 1 + K * lags          # intercept + K vars × lags
    beta0 = np.zeros((n_coef, K))
    V0_diag = np.ones((n_coef, K)) * 1e6   # flat prior on intercept

    if sigma is None:
        sigma = np.ones(K)

    for eq in range(K):
        for lag in range(1, lags + 1):
            for var in range(K):
                coef_idx = 1 + (lag - 1) * K + var
                
                if var == eq:  # Own-lag
                    # Random-walk prior mean = 1 for first lag, 0 for others
                    beta0[coef_idx, eq] = 1.0 if lag == 1 else 0.0
                    v = (lambda1 / (lag ** lambda3)) ** 2
                else:          # Cross-lag
                    # Shrink cross-variable lags scaled by relative residual variance
                    v = (lambda1 * lambda2 / (lag ** lambda3)) ** 2 * (sigma[var] / max(sigma[eq], 1e-8)) ** 2
                
                V0_diag[coef_idx, eq] = v

    return beta0, V0_diag


def _bvar_posterior(X: np.ndarray, Y: np.ndarray, beta0: np.ndarray,
                    V0_diag: np.ndarray) -> np.ndarray:
    """
    Analytically compute BVAR posterior mean equation-by-equation.

    β_post_eq = (X'X + diag(1/V0_diag[:, eq]))⁻¹ (X'Y[:, eq] + diag(1/V0_diag[:, eq]) @ beta0[:, eq])
    """
    n_coef, K = beta0.shape
    beta_post = np.zeros_like(beta0)

    XtX = X.T @ X   # (n_coef, n_coef) — shared across equations

    for eq in range(K):
        v_inv = 1.0 / np.clip(V0_diag[:, eq], 1e-12, None)  # (n_coef,)
        Omega_inv = XtX + np.diag(v_inv)                     # posterior precision
        rhs = X.T @ Y[:, eq] + v_inv * beta0[:, eq]         # (n_coef,)
        try:
            beta_post[:, eq] = np.linalg.solve(Omega_inv, rhs)
        except np.linalg.LinAlgError:
            # Ridge fallback
            beta_post[:, eq] = np.linalg.lstsq(Omega_inv, rhs, rcond=None)[0]

    return beta_post


# ---------------------------------------------------------------------------
# VARNowcast  —  frequentist baseline
# ---------------------------------------------------------------------------

class VARNowcast(BaseNowcastModel):
    """
    Baseline frequentist VAR using statsmodels.
    Equivalent to the old BVARNowcast before the Bayesian upgrade.
    """
    def __init__(self, target_col: str, horizon: int = 1, lags: int = 3, max_vars: int = 30, **kwargs):
        super().__init__(target_col, horizon, **kwargs)
        self.lags = lags
        self.max_vars = max_vars
        self.fitted_res_ = None
        self.endog_names_ = None
        self.n_features_seen_ = 0
        self.prediction_type = "recursive_forecast"

    def _clean(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        full = pd.concat([X, y.rename(self.target_col)], axis=1)
        valid_cols = full.columns[full.isna().mean() < 0.5]
        full = full[valid_cols]
        
        self.n_features_seen_ = X.shape[1]
        max_safe_vars = max(10, min(self.max_vars, max(1, len(full)) // max(1, self.lags)))
        
        if len(full.columns) > max_safe_vars:
            logger.warning(
                f"VAR safeguard: Input has {len(full.columns)} features. "
                f"Truncating to {max_safe_vars} to prevent matrix singularity. "
                f"Use a feature selector before VAR if you see this!"
            )
            completeness = full.notna().mean()
            keep = [self.target_col] + (
                completeness.drop(self.target_col, errors="ignore")
                            .sort_values(ascending=False)
                            .head(max_safe_vars - 1)
                            .index.tolist()
            )
            # Ensure target is still there if dropped
            if self.target_col not in keep: keep = [self.target_col] + keep[:max_safe_vars-1]
            full = full[keep]

        self.endog_names_ = full.columns.tolist()
        return full.ffill().bfill()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        endog = self._clean(X_train, y_train)
        model = VAR(endog)
        # Use fixed lags (not ic='aic') to avoid statsmodels selecting k_ar=0
        lags_to_try = min(self.lags, max(1, (len(endog) - 1) // endog.shape[1]))
        try:
            self.fitted_res_ = model.fit(lags_to_try)
        except Exception as e:
            logger.warning(f"VAR fit failed with {lags_to_try} lags ({e}). Retrying with 1 lag.")
            self.fitted_res_ = model.fit(1)
        self.is_fitted = True
        return self


    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        steps = len(X_test)
        logger.info(f"VAR generating {steps}-step recursive forecast (ignoring X_test values).")
        lag_order = max(self.fitted_res_.k_ar, 1)  # guard: AIC may select 0 lags
        history = self.fitted_res_.endog[-lag_order:]
        try:
            forecast = self.fitted_res_.forecast(history, steps=steps)
        except Exception as e:
            logger.warning(f"VAR forecast failed ({e}). Returning NaN.")
            return pd.Series(np.nan, index=X_test.index, name=f"{self.target_col}_pred")
        fc_df = pd.DataFrame(forecast, index=X_test.index, columns=self.endog_names_)
        if self.target_col in fc_df.columns:
            return fc_df[self.target_col].rename(f"{self.target_col}_pred")
        return pd.Series(np.nan, index=X_test.index, name=f"{self.target_col}_pred")



# ---------------------------------------------------------------------------
# BVARNowcast — true Bayesian VAR with Minnesota Prior
# ---------------------------------------------------------------------------

class BVARNowcast(BaseNowcastModel):
    """
    Bayesian Vector Autoregression with Minnesota shrinkage prior.

    Parameters
    ----------
    mode : str
        'bvar' (default) — Minnesota prior analytical posterior.
        'var'            — delegates to frequentist statsmodels VAR (baseline).
    lags : int
        Lag order. Default 3.
    lambda1 : float
        Overall shrinkage strength. Smaller = more shrinkage. Default 0.1.
    lambda2 : float
        Cross-variable shrinkage relative to own-lag. Default 0.5.
    lambda3 : float
        Lag-decay exponent. Higher = faster decay for longer lags. Default 1.0.
    max_vars : int
        Maximum number of variables to include in the VAR system.
        Set to limit dimensionality (collapses via column selection). Default 30.
    seed : int
        Random seed for reproducibility (used by numpy). Default 123.
    """

    def __init__(
        self,
        target_col: str,
        horizon: int = 1,
        mode: str = "bvar",
        lags: int = 3,
        lambda1: float = 0.1,
        lambda2: float = 0.5,
        lambda3: float = 1.0,
        max_vars: int = 30,
        seed: int = 123,
        **kwargs,
    ):
        super().__init__(target_col, horizon, **kwargs)
        self.mode = mode.lower()
        self.lags = lags
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.max_vars = max_vars
        self.seed = seed

        # Fitted state
        self._beta_post: Optional[np.ndarray] = None    # (1+K*lags, K)
        self._endog_cols: list[str] = []
        self._target_idx: int = -1
        self._lags_used: int = lags
        self.n_features_seen_ = 0
        self.prediction_type = "recursive_forecast"

        # Delegate to baseline VAR if mode == "var"
        self._var_delegate: Optional[VARNowcast] = None
        if self.mode == "var":
            self._var_delegate = VARNowcast(
                target_col=target_col, horizon=horizon, lags=lags, max_vars=max_vars, **kwargs
            )

    # ------------------------------------------------------------------
    def _select_and_clean(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[np.ndarray, list[str], int]:
        """
        Assemble the endogenous panel, drop high-NaN series, cap dimensionality,
        fill remaining NaNs and return numpy array + metadata.
        """
        full = pd.concat([X, y.rename(self.target_col)], axis=1)

        # Drop columns with > 50% missing
        valid_cols = full.columns[full.isna().mean() < 0.5].tolist()
        full = full[valid_cols]

        # Ensure target is included
        if self.target_col not in full.columns:
            raise ValueError(f"Target '{self.target_col}' not present after NaN cleanup.")

        self.n_features_seen_ = X.shape[1]
        
        # Cap dimensionality: Final safety net to avoid singular matrices
        # We allow up to max_vars, but bounded by sample_size / lags
        max_safe_vars = max(10, min(self.max_vars, max(1, len(full)) // max(1, self.lags)))

        if len(full.columns) > max_safe_vars:
            logger.warning(
                f"BVAR safeguard: Input has {len(full.columns)} features. "
                f"Truncating to {max_safe_vars} based on completeness to prevent matrix singularity. "
                f"Use an explicit feature selector before BVAR to avoid this fallback!"
            )
            completeness = full.notna().mean()
            keep = [self.target_col] + (
                completeness.drop(self.target_col)
                            .sort_values(ascending=False)
                            .head(max_safe_vars - 1)
                            .index.tolist()
            )
            full = full[keep]

        full = full.ffill().bfill()

        endog_cols = full.columns.tolist()
        target_idx = endog_cols.index(self.target_col)

        return full.values.astype(float), endog_cols, target_idx

    # ------------------------------------------------------------------
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        np.random.seed(self.seed)

        if self.mode == "var":
            self._var_delegate.fit(X_train, y_train)
            self.is_fitted = True
            return self

        data, endog_cols, target_idx = self._select_and_clean(X_train, y_train)
        T, K = data.shape

        # Reduce lags if not enough data
        lags = min(self.lags, max(1, (T - 1) // K))
        self._lags_used = lags

        X, Y = _build_lag_matrix(data, lags)

        # Per-variable sigma estimate (for cross-variable scaling)
        sigma = np.std(data, axis=0) + 1e-8

        beta0, V0_diag = _minnesota_prior(K, lags, self.lambda1, self.lambda2, self.lambda3, sigma)
        self._beta_post = _bvar_posterior(X, Y, beta0, V0_diag)

        self._endog_cols = endog_cols
        self._target_idx = target_idx
        self._data_train = data   # keep last rows for prediction seed
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        if self.mode == "var":
            return self._var_delegate.predict(X_test)

        lags = self._lags_used
        K = len(self._endog_cols)
        steps = len(X_test)
        logger.info(f"BVAR generating {steps}-step recursive forecast (ignoring X_test values).")

        # Seed the history with the last `lags` rows from training
        history = self._data_train[-lags:].copy()   # (lags, K)

        forecasts = []
        for _ in range(steps):
            # Build single-row X
            x_row = [1.0]   # intercept
            for lag in range(1, lags + 1):
                x_row.extend(history[-lag, :].tolist())
            x_row_arr = np.array(x_row).reshape(1, -1)   # (1, 1+K*lags)

            y_hat = x_row_arr @ self._beta_post           # (1, K)
            forecasts.append(y_hat[0, self._target_idx])

            # Append predicted row to history for multi-step auto-forecast
            history = np.vstack([history, y_hat])

        return pd.Series(forecasts, index=X_test.index, name=f"{self.target_col}_pred")
