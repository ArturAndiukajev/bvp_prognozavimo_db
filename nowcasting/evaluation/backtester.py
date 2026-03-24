"""
backtester.py — Rolling Backtest Evaluator
==========================================
Evaluates nowcast models over a rolling time window while strictly preventing
data leakage (transformer and model are only ever fit on the training window).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("nowcast_backtest")


from sklearn.base import clone

class RollingBacktester:
    """
    Rolling-window backtester.

    At each step t:
        1. Train window  : rows 0 … t-1
        2. Test window   : rows t … t + step_size - 1
        3. Fit transformer on Train, apply to Train + Test
        4. Fit feature_selector on Train, apply to Train + Test
        5. Fit model on Train
        6. Predict Test

    Parameters
    ----------
    initial_train_periods : int
        Minimum number of rows in the first training window.
    step_size : int
        Number of periods to advance the window each iteration.
    min_target_obs : int
        Minimum number of non-NaN target observations required in the
        training window before fitting the model.
    eval_mode : str
        'common_frequency' (default) restricts predictions to X_test.index.
        'mixed_frequency' allows models to return predictions on low-frequency dates.
    window_type : str
        'expanding' (default) uses all data from the start of the panel.
        'rolling' uses a fixed window of size `initial_train_periods`.
    """

    def __init__(
        self,
        initial_train_periods: int = 120,
        step_size: int = 1,
        min_target_obs: int = 8,
        eval_mode: str = "common_frequency",
        window_type: str = "expanding",
    ):
        self.initial_train_periods = initial_train_periods
        self.step_size = step_size
        self.min_target_obs = min_target_obs
        self.eval_mode = eval_mode
        self.window_type = window_type

    # ------------------------------------------------------------------
    def backtest(
        self,
        model,
        X_panel: pd.DataFrame,
        y_target: pd.Series,
        transformer=None,
        feature_selector=None,
    ) -> pd.DataFrame:
        """
        Run the rolling backtest.

        Parameters
        ----------
        model : BaseNowcastModel
        X_panel : pd.DataFrame  (features)
        y_target : pd.Series    (target)
        transformer : optional sklearn-like transformer (fit on X_train)
        feature_selector : optional sklearn-like selector (fit on X_train)

        Returns
        -------
        pd.DataFrame with columns ['Actual', 'Predicted']
        """
        # Align X and y on a common index
        common_idx = X_panel.index.intersection(y_target.index)
        if len(common_idx) == 0:
            logger.error(
                f"Backtester: X_panel and y_target have NO shared index dates. "
                f"X range=[{X_panel.index.min()}..{X_panel.index.max()}], "
                f"y range=[{y_target.index.min()}..{y_target.index.max()}]"
            )
            return pd.DataFrame()

        if len(common_idx) < len(X_panel):
            logger.info(
                f"Backtester: aligning X ({len(X_panel)}) / y ({len(y_target)}) "
                f"→ {len(common_idx)} common dates."
            )
            X_panel = X_panel.loc[common_idx]
            y_target = y_target.loc[common_idx]

        out_preds = []
        n_total = len(X_panel)
        n_steps_run = 0
        n_steps_failed = 0

        logger.info(
            f"Starting backtest: {n_total} periods | "
            f"initial_train={self.initial_train_periods} | step_size={self.step_size}"
        )

        for t in range(self.initial_train_periods, n_total, self.step_size):
            train_end = t
            test_end = min(t + self.step_size, n_total)
            
            if self.window_type == "rolling":
                train_start = max(0, train_end - self.initial_train_periods)
            else:
                train_start = 0

            X_train_orig = X_panel.iloc[train_start:train_end].copy()
            y_train = y_target.iloc[train_start:train_end].copy()
            X_test_orig = X_panel.iloc[train_end:test_end].copy()
            
            X_train = X_train_orig
            X_test = X_test_orig
            
            n_raw_features = X_train.shape[1]
            n_trans_features = n_raw_features
            n_sel_features = n_raw_features
            
            forecast_origin = X_train.index[-1] if not X_train.empty else pd.NaT

            # Guard: need enough non-NaN target observations to train
            if y_train.notna().sum() < self.min_target_obs:
                logger.debug(
                    f"  step t={t}: skipping — only {y_train.notna().sum()} "
                    f"target obs (< {self.min_target_obs})"
                )
                out_preds.append(pd.Series(np.nan, index=X_test.index))
                continue

            # 1. Optional transform
            if transformer is not None:
                try:
                    step_trans = clone(transformer)
                    step_trans.fit(X_train, y_train)
                    X_train = step_trans.transform(X_train)
                    X_test = step_trans.transform(X_test)
                    n_trans_features = X_train.shape[1]
                except Exception as e:
                    logger.warning(f"  step t={t}: transformer failed: {e}")

            # 2. Optional feature selection
            if feature_selector is not None:
                try:
                    step_sel = clone(feature_selector)
                    step_sel.fit(X_train, y_train)
                    X_train = step_sel.transform(X_train)
                    X_test = step_sel.transform(X_test)
                    n_sel_features = X_train.shape[1]
                except Exception as e:
                    logger.warning(f"  step t={t}: feature_selector failed: {e}")

            # 3. Fit + predict
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                if isinstance(preds, pd.DataFrame):
                    if self.eval_mode == "common_frequency":
                        if not preds.index.equals(X_test.index):
                            preds = preds.reindex(X_test.index)
                    else:
                        # Contract enforcement: Mixed-frequency mode must use the target horizon indices explicitly defined by the model
                        if preds.empty:
                            logger.warning(f"  step t={t}: mixed-frequency prediction returned empty DataFrame.")
                    preds_df = preds.rename(columns={preds.columns[0]: "Predicted"})
                else:
                    if self.eval_mode == "common_frequency":
                        if not preds.index.equals(X_test.index):
                            preds = preds.reindex(X_test.index)
                    else:
                        if preds.empty:
                            logger.warning(f"  step t={t}: mixed-frequency prediction returned empty Series.")
                    preds_df = preds.to_frame(name="Predicted")
                
                preds_df["Target_Date"] = preds_df.index
                preds_df["Forecast_Origin"] = forecast_origin
                preds_df["Model"] = model.__class__.__name__
                preds_df["Prediction_Type"] = getattr(model, "prediction_type", "conditional_nowcast")
                preds_df["n_raw_features"] = n_raw_features
                preds_df["n_trans_features"] = n_trans_features
                preds_df["n_sel_features"] = n_sel_features
                if hasattr(model, "n_features_seen_"):
                    preds_df["n_model_used_features"] = model.n_features_seen_
                else:
                    preds_df["n_model_used_features"] = n_sel_features
                
                out_preds.append(preds_df)
                n_steps_run += 1
            except Exception as e:
                logger.error(
                    f"  step {n_steps_run + n_steps_failed + 1} | {model.__class__.__name__} failed: {e} | "
                    f"X_train={X_train.shape}, X_test={X_test.shape}, "
                    f"origin={forecast_origin.date() if isinstance(forecast_origin, pd.Timestamp) else forecast_origin}"
                )
                
                # Append empty dataframe with expected schema to preserve tracking
                err_df = pd.DataFrame(index=X_test.index)
                err_df["Predicted"] = np.nan
                err_df["Target_Date"] = err_df.index
                err_df["Forecast_Origin"] = forecast_origin
                err_df["Model"] = model.__class__.__name__
                out_preds.append(err_df)
                
                n_steps_failed += 1

        logger.info(
            f"Backtest complete: {n_steps_run} steps succeeded, "
            f"{n_steps_failed} failed."
        )

        if not out_preds:
            return pd.DataFrame()

        combined_preds = pd.concat(out_preds)
        
        # Reset index instead of relying on purely index-based alignment
        combined_preds = combined_preds.reset_index(drop=True)
        
        # Pull actuals using the target dates
        combined_preds["Actual"] = combined_preds["Target_Date"].map(y_target)
        
        # Reorder and filter columns for evaluation
        cols = ["Forecast_Origin", "Target_Date", "Model", "Prediction_Type", "Actual", "Predicted"]
        eval_df = combined_preds[cols].sort_values(by=["Forecast_Origin", "Target_Date"]).reset_index(drop=True)
        
        return eval_df
