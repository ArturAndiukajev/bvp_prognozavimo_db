"""
backtester.py — Rolling / Expanding Backtest Evaluator
======================================================
Evaluates nowcast models over a rolling or expanding time window while strictly
preventing data leakage (transformers and models are only fit on the training
window).

Supports:
- common_frequency: X and y share the same cadence/index grid
- mixed_frequency: intended for bridge-style models where X is high-frequency
  and y is low-frequency; iteration is done over observed target dates
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone

logger = logging.getLogger("nowcast_backtest")


class RollingBacktester:
    """
    Backtester for nowcasting models.

    Parameters
    ----------
    initial_train_periods : int
        Minimum number of observations in the first training window.
        In common_frequency mode this counts X/y rows.
        In mixed_frequency mode this counts observed low-frequency target points.
    step_size : int
        Number of periods to advance the window each iteration.
        In mixed_frequency mode this advances by low-frequency target dates.
    min_target_obs : int
        Minimum number of non-NaN target observations required in the
        training window before fitting the model.
    eval_mode : str
        'common_frequency' or 'mixed_frequency'.
    window_type : str
        'expanding' or 'rolling'.
    rolling_window_size : int | None
        Window size for rolling mode.
        In common_frequency mode this counts X/y rows.
        In mixed_frequency mode this counts low-frequency target observations.
    """

    def __init__(
        self,
        initial_train_periods: int = 120,
        step_size: int = 1,
        min_target_obs: int = 8,
        eval_mode: str = "common_frequency",
        window_type: str = "expanding",
        rolling_window_size: Optional[int] = None,
    ):
        self.initial_train_periods = int(initial_train_periods)
        self.step_size = int(step_size)
        self.min_target_obs = int(min_target_obs)
        self.eval_mode = eval_mode
        self.window_type = window_type
        self.rolling_window_size = rolling_window_size

    def backtest(
        self,
        model,
        X_panel: pd.DataFrame,
        y_target: pd.Series,
        transformer=None,
        feature_selector=None,
    ) -> pd.DataFrame:
        if X_panel is None or y_target is None or X_panel.empty or y_target.empty:
            logger.warning("Backtester: empty X_panel or y_target.")
            return pd.DataFrame()

        if self.eval_mode == "mixed_frequency":
            return self._backtest_mixed_frequency(
                model=model,
                X_panel=X_panel,
                y_target=y_target,
                transformer=transformer,
                feature_selector=feature_selector,
            )

        return self._backtest_common_frequency(
            model=model,
            X_panel=X_panel,
            y_target=y_target,
            transformer=transformer,
            feature_selector=feature_selector,
        )

    # ------------------------------------------------------------------
    # Common-frequency path
    # ------------------------------------------------------------------
    def _backtest_common_frequency(
        self,
        model,
        X_panel: pd.DataFrame,
        y_target: pd.Series,
        transformer=None,
        feature_selector=None,
    ) -> pd.DataFrame:
        common_idx = X_panel.index.intersection(y_target.index)
        if len(common_idx) == 0:
            logger.error(
                "Backtester: X_panel and y_target have no shared index dates. "
                f"X range=[{X_panel.index.min()}..{X_panel.index.max()}], "
                f"y range=[{y_target.index.min()}..{y_target.index.max()}]"
            )
            return pd.DataFrame()

        if len(common_idx) < len(X_panel) or len(common_idx) < len(y_target):
            logger.info(
                f"Backtester: aligning X ({len(X_panel)}) / y ({len(y_target)}) "
                f"→ {len(common_idx)} common dates."
            )
            X_panel = X_panel.loc[common_idx].copy()
            y_target = y_target.loc[common_idx].copy()

        out_preds = []
        n_total = len(X_panel)
        n_steps_run = 0
        n_steps_failed = 0

        logger.info(
            f"Starting CF backtest: {n_total} periods | "
            f"initial_train={self.initial_train_periods} | step_size={self.step_size} | "
            f"window_type={self.window_type}"
        )

        for t in range(self.initial_train_periods, n_total, self.step_size):
            train_end = t
            test_end = min(t + self.step_size, n_total)

            if self.window_type == "rolling":
                rw = int(self.rolling_window_size) if self.rolling_window_size is not None else int(self.initial_train_periods)
                train_start = max(0, train_end - rw)
            else:
                train_start = 0

            X_train_orig = X_panel.iloc[train_start:train_end].copy()
            y_train = y_target.iloc[train_start:train_end].copy()
            X_test_orig = X_panel.iloc[train_end:test_end].copy()

            forecast_origin = X_train_orig.index[-1] if not X_train_orig.empty else pd.NaT
            n_raw_features = X_train_orig.shape[1]
            n_trans_features = n_raw_features
            n_sel_features = n_raw_features

            if y_train.notna().sum() < self.min_target_obs:
                logger.debug(
                    f"step t={t}: skipping — only {y_train.notna().sum()} "
                    f"target obs (< {self.min_target_obs})"
                )
                out_preds.append(
                    self._empty_prediction_frame(
                        target_index=X_test_orig.index,
                        forecast_origin=forecast_origin,
                        model_name=model.__class__.__name__,
                        prediction_type=getattr(model, "prediction_type", "conditional_nowcast"),
                        n_raw_features=n_raw_features,
                        n_trans_features=n_trans_features,
                        n_sel_features=n_sel_features,
                        n_model_used_features=n_sel_features,
                    )
                )
                continue

            X_train = X_train_orig
            X_test = X_test_orig

            if transformer is not None:
                X_train, X_test, n_trans_features = self._apply_step(
                    step=transformer,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    step_name="transformer",
                    default_prefix="trans",
                )

            if feature_selector is not None:
                X_train, X_test, n_sel_features = self._apply_step(
                    step=feature_selector,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    step_name="feature_selector",
                    default_prefix="feat",
                )

            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                preds_df = self._coerce_predictions(
                    preds=preds,
                    target_index=X_test.index,
                    eval_mode="common_frequency",
                )
                preds_df["Forecast_Origin"] = forecast_origin
                preds_df["Model"] = model.__class__.__name__
                preds_df["Prediction_Type"] = getattr(model, "prediction_type", "conditional_nowcast")
                preds_df["n_raw_features"] = n_raw_features
                preds_df["n_trans_features"] = n_trans_features
                preds_df["n_sel_features"] = n_sel_features
                preds_df["n_model_used_features"] = getattr(model, "n_features_seen_", n_sel_features)

                out_preds.append(preds_df)
                n_steps_run += 1
            except Exception as e:
                logger.error(
                    f"CF step failed | {model.__class__.__name__}: {e} | "
                    f"X_train={X_train.shape}, X_test={X_test.shape}, "
                    f"origin={forecast_origin}"
                )
                out_preds.append(
                    self._empty_prediction_frame(
                        target_index=X_test_orig.index,
                        forecast_origin=forecast_origin,
                        model_name=model.__class__.__name__,
                        prediction_type=getattr(model, "prediction_type", "conditional_nowcast"),
                        n_raw_features=n_raw_features,
                        n_trans_features=n_trans_features,
                        n_sel_features=n_sel_features,
                        n_model_used_features=n_sel_features,
                    )
                )
                n_steps_failed += 1

        logger.info(
            f"CF backtest complete: {n_steps_run} steps succeeded, {n_steps_failed} failed."
        )
        return self._finalize_predictions(out_preds, y_target)

    # ------------------------------------------------------------------
    # Mixed-frequency path (Bridge-style)
    # ------------------------------------------------------------------
    def _backtest_mixed_frequency(
        self,
        model,
        X_panel: pd.DataFrame,
        y_target: pd.Series,
        transformer=None,
        feature_selector=None,
    ) -> pd.DataFrame:
        # Mixed-frequency contract:
        # - X_panel is high-frequency
        # - y_target is low-frequency
        # - iterate over observed target dates, not HF row positions
        target_obs = y_target.dropna().sort_index()
        if target_obs.empty:
            logger.error("Backtester MF: y_target has no observed dates.")
            return pd.DataFrame()

        if len(target_obs) <= self.initial_train_periods:
            logger.warning(
                f"Backtester MF: not enough observed target dates ({len(target_obs)}) "
                f"for initial_train_periods={self.initial_train_periods}."
            )
            return pd.DataFrame()

        out_preds = []
        n_steps_run = 0
        n_steps_failed = 0

        logger.info(
            f"Starting MF backtest: {len(target_obs)} observed target dates | "
            f"initial_train={self.initial_train_periods} | step_size={self.step_size} | "
            f"window_type={self.window_type}"
        )

        for pos in range(self.initial_train_periods, len(target_obs), self.step_size):
            pred_target_dates = target_obs.index[pos : pos + self.step_size]
            if len(pred_target_dates) == 0:
                continue

            if self.window_type == "rolling":
                rw = int(self.rolling_window_size) if self.rolling_window_size is not None else int(self.initial_train_periods)
                train_target_dates = target_obs.index[max(0, pos - rw) : pos]
            else:
                train_target_dates = target_obs.index[:pos]

            if len(train_target_dates) == 0:
                continue

            train_target_end = train_target_dates[-1]
            test_target_end = pred_target_dates[-1]

            # HF train history up to the last training LF target date
            X_train_orig = X_panel.loc[:train_target_end].copy()
            y_train = y_target.loc[train_target_dates].copy()

            # HF test slice after train end and up to the last predicted LF target date
            X_test_orig = X_panel.loc[(X_panel.index > train_target_end) & (X_panel.index <= test_target_end)].copy()

            # Ensure predict() always receives a non-empty frame with a valid end_date
            if X_test_orig.empty:
                fallback_test = X_panel.loc[:test_target_end].tail(1).copy()
                X_test_orig = fallback_test

            forecast_origin = X_train_orig.index[-1] if not X_train_orig.empty else pd.NaT
            n_raw_features = X_train_orig.shape[1]
            n_trans_features = n_raw_features
            n_sel_features = n_raw_features

            if X_train_orig.empty or y_train.notna().sum() < self.min_target_obs:
                logger.debug(
                    f"MF step at target={test_target_end}: skipping — "
                    f"train_hf_rows={len(X_train_orig)}, train_target_obs={y_train.notna().sum()}"
                )
                out_preds.append(
                    self._empty_prediction_frame(
                        target_index=pred_target_dates,
                        forecast_origin=forecast_origin,
                        model_name=model.__class__.__name__,
                        prediction_type=getattr(model, "prediction_type", "conditional_nowcast"),
                        n_raw_features=n_raw_features,
                        n_trans_features=n_trans_features,
                        n_sel_features=n_sel_features,
                        n_model_used_features=n_sel_features,
                    )
                )
                continue

            X_train = X_train_orig
            X_test = X_test_orig

            # For mixed-frequency bridge workflows, supervised external selectors
            # often do not apply cleanly because HF X and LF y differ in length.
            # We therefore try y-aware fit only when lengths match; otherwise we
            # fall back to unsupervised fit(X). If that fails, the step is skipped.
            if transformer is not None:
                X_train, X_test, n_trans_features = self._apply_step(
                    step=transformer,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    step_name="transformer",
                    default_prefix="trans",
                )

            if feature_selector is not None:
                X_train, X_test, n_sel_features = self._apply_step(
                    step=feature_selector,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    step_name="feature_selector",
                    default_prefix="feat",
                )

            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                preds_df = self._coerce_predictions(
                    preds=preds,
                    target_index=pred_target_dates,
                    eval_mode="mixed_frequency",
                )
                preds_df["Forecast_Origin"] = forecast_origin
                preds_df["Model"] = model.__class__.__name__
                preds_df["Prediction_Type"] = getattr(model, "prediction_type", "conditional_nowcast")
                preds_df["n_raw_features"] = n_raw_features
                preds_df["n_trans_features"] = n_trans_features
                preds_df["n_sel_features"] = n_sel_features
                preds_df["n_model_used_features"] = getattr(model, "n_features_seen_", n_sel_features)

                out_preds.append(preds_df)
                n_steps_run += 1
            except Exception as e:
                logger.error(
                    f"MF step failed | {model.__class__.__name__}: {e} | "
                    f"X_train={X_train.shape}, X_test={X_test.shape}, "
                    f"forecast_origin={forecast_origin}, target_end={test_target_end}"
                )
                out_preds.append(
                    self._empty_prediction_frame(
                        target_index=pred_target_dates,
                        forecast_origin=forecast_origin,
                        model_name=model.__class__.__name__,
                        prediction_type=getattr(model, "prediction_type", "conditional_nowcast"),
                        n_raw_features=n_raw_features,
                        n_trans_features=n_trans_features,
                        n_sel_features=n_sel_features,
                        n_model_used_features=n_sel_features,
                    )
                )
                n_steps_failed += 1

        logger.info(
            f"MF backtest complete: {n_steps_run} steps succeeded, {n_steps_failed} failed."
        )
        return self._finalize_predictions(out_preds, y_target)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _apply_step(
        self,
        step,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        step_name: str,
        default_prefix: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        try:
            fitted = clone(step)

            # y-aware fit only when lengths match
            if y_train is not None and len(y_train) == len(X_train):
                try:
                    fitted.fit(X_train, y_train)
                except TypeError:
                    fitted.fit(X_train)
            else:
                fitted.fit(X_train)

            X_train_t = fitted.transform(X_train)
            X_test_t = fitted.transform(X_test)

            X_train_df = self._to_frame_like(X_train_t, X_train.index, prefix=default_prefix)
            X_test_df = self._to_frame_like(X_test_t, X_test.index, prefix=default_prefix, columns=X_train_df.columns)
            return X_train_df, X_test_df, X_train_df.shape[1]
        except Exception as e:
            logger.warning(f"{step_name} failed and will be skipped: {e}")
            return X_train, X_test, X_train.shape[1]

    @staticmethod
    def _to_frame_like(obj, index, prefix: str, columns=None) -> pd.DataFrame:
        if isinstance(obj, pd.DataFrame):
            out = obj.copy()
            out.index = index
            if columns is not None and len(columns) == out.shape[1]:
                out.columns = columns
            return out

        if isinstance(obj, pd.Series):
            name = columns[0] if columns is not None and len(columns) == 1 else prefix
            return obj.rename(name).to_frame().set_index(pd.Index(index))

        arr = np.asarray(obj)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if columns is None:
            columns = [f"{prefix}_{i}" for i in range(arr.shape[1])]

        return pd.DataFrame(arr, index=index, columns=list(columns))

    @staticmethod
    def _coerce_predictions(preds, target_index, eval_mode: str) -> pd.DataFrame:
        if isinstance(preds, pd.DataFrame):
            out = preds.copy()
            if "Predicted" not in out.columns:
                first_col = out.columns[0]
                out = out.rename(columns={first_col: "Predicted"})
            out = out[["Predicted"]]
        else:
            out = preds.to_frame(name="Predicted")

        if eval_mode == "common_frequency":
            if not out.index.equals(target_index):
                out = out.reindex(target_index)
        else:
            # In mixed-frequency mode we evaluate on explicit LF target dates.
            out = out.reindex(target_index)

        out["Target_Date"] = out.index
        return out

    @staticmethod
    def _empty_prediction_frame(
        target_index,
        forecast_origin,
        model_name,
        prediction_type,
        n_raw_features,
        n_trans_features,
        n_sel_features,
        n_model_used_features,
    ) -> pd.DataFrame:
        df = pd.DataFrame(index=pd.Index(target_index))
        df["Predicted"] = np.nan
        df["Target_Date"] = df.index
        df["Forecast_Origin"] = forecast_origin
        df["Model"] = model_name
        df["Prediction_Type"] = prediction_type
        df["n_raw_features"] = n_raw_features
        df["n_trans_features"] = n_trans_features
        df["n_sel_features"] = n_sel_features
        df["n_model_used_features"] = n_model_used_features
        return df

    @staticmethod
    def _finalize_predictions(out_preds, y_target: pd.Series) -> pd.DataFrame:
        if not out_preds:
            return pd.DataFrame()

        combined = pd.concat(out_preds, axis=0)
        combined = combined.reset_index(drop=True)
        combined["Actual"] = combined["Target_Date"].map(y_target)

        cols = [
            "Forecast_Origin",
            "Target_Date",
            "Model",
            "Prediction_Type",
            "Actual",
            "Predicted",
            "n_raw_features",
            "n_trans_features",
            "n_sel_features",
            "n_model_used_features",
        ]

        for col in cols:
            if col not in combined.columns:
                combined[col] = np.nan

        return combined[cols].sort_values(
            by=["Forecast_Origin", "Target_Date"]
        ).reset_index(drop=True)