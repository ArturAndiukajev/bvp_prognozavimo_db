"""
dfm.py — Dynamic Factor Model (Common-Frequency & Mixed-Frequency)
===================================================================

Common-frequency mode  (mixed_frequency=False):
    Uses statsmodels DynamicFactor.  Handles ragged edges via Kalman Filter.

Mixed-frequency mode  (mixed_frequency=True):
    Uses statsmodels DynamicFactorMQ, which natively supports a mixture of
    monthly and quarterly series in the same state-space model.  This is the
    standard ECB/Fed approach to mixed-frequency nowcasting.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from nowcasting.models.base import BaseNowcastModel

logger = logging.getLogger("nowcast_dfm")


class DynamicFactorNowcast(BaseNowcastModel):
    """
    Dynamic Factor Model wrapper.

    Parameters
    ----------
    target_col : str
    horizon : int
    k_factors : int
        Number of latent factors.
    factor_order : int
        AR order of the factor process.
    mixed_frequency : bool
        If True, uses DynamicFactorMQ (monthly + quarterly mixed).
        If False, uses DynamicFactor (single common frequency).
    quarterly_cols : list[str] | None
        Column names that are at quarterly frequency (only used when
        mixed_frequency=True). All other columns are treated as monthly.
    """

    def __init__(
        self,
        target_col: str,
        horizon: int = 1,
        k_factors: int = 2,
        factor_order: int = 2,
        mixed_frequency: bool = False,
        quarterly_cols: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(target_col, horizon, **kwargs)
        self.k_factors = k_factors
        self.factor_order = factor_order
        self.mixed_frequency = mixed_frequency
        self.quarterly_cols = quarterly_cols or []
        self.fitted_res_ = None

    # ------------------------------------------------------------------
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "DynamicFactorNowcast":
        endog = pd.concat([X_train, y_train.rename(self.target_col)], axis=1)
        endog = endog.dropna(axis=1, how="all")

        if endog.shape[1] < 2:
            logger.warning("DFM: fewer than 2 series after dropping all-NaN columns — may fail.")

        if self.mixed_frequency:
            self._fit_mq(endog)
        else:
            self._fit_standard(endog)

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def _fit_standard(self, endog: pd.DataFrame) -> None:
        """Common-frequency DFM via statsmodels DynamicFactor."""
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

        model = DynamicFactor(
            endog,
            k_factors=min(self.k_factors, endog.shape[1] - 1),
            factor_order=self.factor_order,
            enforce_stationarity=False,
        )
        logger.info(
            f"[DFM-CF] Fitting DynamicFactor ({self.k_factors} factors, AR({self.factor_order})) "
            f"on {endog.shape}..."
        )
        try:
            self.fitted_res_ = model.fit(method="powell", disp=False)
        except Exception as e:
            logger.warning(f"[DFM-CF] Powell failed: {e}. Falling back to EM.")
            self.fitted_res_ = model.fit(method="em", disp=False, maxiter=100)

    # ------------------------------------------------------------------
    def _fit_mq(self, endog: pd.DataFrame) -> None:
        """
        Mixed-frequency DFM via statsmodels DynamicFactorMQ.

        DynamicFactorMQ requires:
         - A monthly frequency index (or quarterly, but mixed is handled by the model)
         - A dict `endog_quarterly` specifying which columns are observed quarterly
        """
        try:
            from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
        except ImportError:
            logger.error(
                "[DFM-MF] DynamicFactorMQ not available — requires statsmodels ≥ 0.13. "
                "Falling back to common-frequency DFM."
            )
            self._fit_standard(endog)
            return

        # Separate monthly and quarterly columns
        q_cols = [c for c in self.quarterly_cols if c in endog.columns]
        m_cols = [c for c in endog.columns if c not in q_cols]

        logger.info(
            f"[DFM-MF] Fitting DynamicFactorMQ: {len(m_cols)} monthly, "
            f"{len(q_cols)} quarterly, "
            f"{self.k_factors} factors..."
        )

        endog_m = endog[m_cols] if m_cols else endog
        endog_q = endog[q_cols] if q_cols else None

        try:
            model = DynamicFactorMQ(
                endog_m,
                endog_quarterly=endog_q,
                factors=self.k_factors,
                factor_orders=self.factor_order,
                idiosyncratic_ar1=True,
            )
            self.fitted_res_ = model.fit(
                disp=False,
                maxiter=200,
                method="em",
            )
        except Exception as e:
            logger.error(f"[DFM-MF] DynamicFactorMQ failed: {e}. Falling back to CF-DFM.")
            self._fit_standard(endog)

    # ------------------------------------------------------------------
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("DFM not fitted.")

        if self.mixed_frequency:
            return self._predict_mq(X_test)
        else:
            return self._predict_standard(X_test)

    # ------------------------------------------------------------------
    def _predict_standard(self, X_test: pd.DataFrame) -> pd.Series:
        """Extend Kalman Filter through the test period (CF mode)."""
        y_dummy = pd.Series(np.nan, index=X_test.index, name=self.target_col)
        test_endog = pd.concat([X_test, y_dummy], axis=1)

        try:
            endog_names = self.fitted_res_.model.endog_names
            # Keep only columns that were in the fitted model
            test_endog = test_endog.reindex(columns=endog_names)
        except Exception:
            pass

        try:
            # TRUE CONDITIONAL NOWCASTING
            # We append the test period observations (where predictors are known but target is NaN).
            # The Kalman Filter will smooth through the new observations and generate an optimal target estimate.
            extended_res = self.fitted_res_.append(test_endog, refit=False)
            
            # The appended portion contains our nowcast (fittedvalues for the target col)
            predictions = extended_res.fittedvalues[self.target_col].iloc[-len(X_test):]
            return pd.Series(predictions.values, index=X_test.index, name=f"{self.target_col}_pred")
        except Exception as e:
            logger.error(f"[DFM-CF] predict append failed: {e}. Falling back to unconditional forecast.")
            try:
                forecast = self.fitted_res_.forecast(steps=len(X_test))
                if isinstance(forecast, pd.DataFrame):
                    return forecast[self.target_col]
                else:
                    return pd.Series(forecast, index=X_test.index, name=f"{self.target_col}_pred")
            except Exception as e2:
                logger.error(f"[DFM-CF] unconditional predict failed: {e2}")
                return pd.Series(np.nan, index=X_test.index, name=f"{self.target_col}_pred")

    # ------------------------------------------------------------------
    def _predict_mq(self, X_test: pd.DataFrame) -> pd.Series:
        """Extend Kalman Filter through the test period (MF mode)."""
        y_dummy = pd.Series(np.nan, index=X_test.index, name=self.target_col)
        test_endog = pd.concat([X_test, y_dummy], axis=1)

        try:
            endog_names = self.fitted_res_.model.endog_names
            test_endog = test_endog.reindex(columns=endog_names)
        except Exception:
            pass

        try:
            # TRUE CONDITIONAL NOWCASTING for Mixed Frequency
            # We specify which columns are monthly vs quarterly, identically to how it was fit.
            q_cols = [c for c in self.quarterly_cols if c in test_endog.columns]
            m_cols = [c for c in test_endog.columns if c not in q_cols]
            
            test_endog_m = test_endog[m_cols] if m_cols else test_endog
            test_endog_q = test_endog[q_cols] if q_cols else None
            
            # append allows extending the state with new test obs
            extended_res = self.fitted_res_.append(
                endog=test_endog_m,
                endog_quarterly=test_endog_q,
                refit=False
            )
            
            predictions = extended_res.fittedvalues[self.target_col].iloc[-len(X_test):]
            return pd.Series(predictions.values, index=X_test.index, name=f"{self.target_col}_pred")
            
        except Exception as e:
            logger.warning(f"[DFM-MF] predict append failed: {e}. Falling back to unconditional forecast.")
            try:
                n_steps = len(X_test)
                forecast = self.fitted_res_.forecast(steps=n_steps)

                if isinstance(forecast, pd.DataFrame) and self.target_col in forecast.columns:
                    return pd.Series(
                        forecast[self.target_col].values,
                        index=X_test.index[:n_steps],
                        name=f"{self.target_col}_pred",
                    )
                return pd.Series(
                    forecast.iloc[:, 0].values,
                    index=X_test.index[:n_steps],
                    name=f"{self.target_col}_pred",
                )
            except Exception as e2:
                logger.error(f"[DFM-MF] predict fallback failed: {e2}")
                return pd.Series(np.nan, index=X_test.index, name=f"{self.target_col}_pred")
