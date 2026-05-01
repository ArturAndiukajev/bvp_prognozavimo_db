import os
import logging
import time
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

try:
    from gluonts.dataset.common import ListDataset
    from tactis.gluon.estimator import TACTiSEstimator
    from tactis.gluon.trainer import TACTISTrainer
    TRACTIS_AVAILABLE = True
except ImportError:
    TRACTIS_AVAILABLE = False

logger = logging.getLogger("tactis2_filler")

class TACTiS2Filler:
    """
    Multivariate predictor filling using TACTiS-2.
    Acting as a ragged-edge filler, not the final forecaster.
    """
    def __init__(
        self,
        author_config: bool = False,
        context_length: int = 120,
        prediction_length: int = 6,
        max_epochs: int = 20,
        epochs_phase_1: int = 20,
        epochs_phase_2: int = 20,
        batch_size: Optional[int] = None,
        num_batches_per_epoch: Optional[int] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        maximum_learning_rate: float = 1e-3,
        clip_gradient: float = 1e3,
        bagging_size: Optional[int] = None,
        skip_copula: Optional[bool] = None,
        num_samples: Optional[int] = None,
        device: str = "auto",
        cache_dir: str = "data/cache/tactis2",
        output_dir: str = "data/forecasts/tactis2_filled_values",
        panel_out_dir: str = "data/forecasts/tactis2_filled_panels",
        checkpoint_dir: Optional[str] = None,
        random_state: int = 2234,
    ):
        self.author_config = author_config
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.max_epochs = max_epochs
        self.epochs_phase_1 = epochs_phase_1
        self.epochs_phase_2 = epochs_phase_2
        
        # Apply mode-specific defaults
        if self.author_config:
            self.batch_size = batch_size if batch_size is not None else 64
            self.num_batches_per_epoch = num_batches_per_epoch if num_batches_per_epoch is not None else 64
            self.bagging_size = bagging_size if bagging_size is not None else 20
            self.skip_copula = skip_copula if skip_copula is not None else False
            self.num_samples = num_samples if num_samples is not None else 100
        else:
            self.batch_size = batch_size if batch_size is not None else 32
            self.num_batches_per_epoch = num_batches_per_epoch if num_batches_per_epoch is not None else 32
            self.bagging_size = bagging_size # default None
            self.skip_copula = skip_copula if skip_copula is not None else True
            self.num_samples = num_samples if num_samples is not None else 20
            
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.maximum_learning_rate = maximum_learning_rate
        self.clip_gradient = clip_gradient
        self.random_state = random_state
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.panel_out_dir = Path(panel_out_dir)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.cache_dir / "checkpoints"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.panel_out_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Default model parameters
        if self.author_config:
            self.model_parameters = {
                "flow_series_embedding_dim": 5,
                "copula_series_embedding_dim": 5,
                "flow_input_encoder_layers": 2,
                "copula_input_encoder_layers": 2,
                "input_encoding_normalization": True,
                "data_normalization": "standardization",
                "loss_normalization": "series",
                "bagging_size": self.bagging_size if self.bagging_size is not None else 20,
                "positional_encoding": {"dropout": 0.0},
                "flow_temporal_encoder": {
                    "attention_layers": 2,
                    "attention_heads": 1,
                    "attention_dim": 16,
                    "attention_feedforward_dim": 16,
                    "dropout": 0.0,
                },
                "copula_temporal_encoder": {
                    "attention_layers": 2,
                    "attention_heads": 1,
                    "attention_dim": 16,
                    "attention_feedforward_dim": 16,
                    "dropout": 0.0,
                },
                "copula_decoder": {
                    "min_u": 0.05,
                    "max_u": 0.95,
                    "attentional_copula": {
                        "attention_heads": 3,
                        "attention_layers": 1,
                        "attention_dim": 8,
                        "mlp_layers": 2,
                        "mlp_dim": 48,
                        "resolution": 20,
                        "activation_function": "relu",
                    },
                    "dsf_marginal": {
                        "mlp_layers": 2,
                        "mlp_dim": 48,
                        "flow_layers": 2,
                        "flow_hid_dim": 8,
                    },
                },
                "experiment_mode": "forecasting",
                "skip_copula": False,
            }
        else:
            # Simplified marginal-only version
            self.model_parameters = {
                "flow_series_embedding_dim": 5,
                "copula_series_embedding_dim": 5,
                "flow_input_encoder_layers": 2,
                "copula_input_encoder_layers": 2,
                "bagging_size": self.bagging_size, # Disable bagging for small-scale filling
                "input_encoding_normalization": True,
                "data_normalization": "standardization",
                "loss_normalization": "both",
                "positional_encoding": {"dropout": 0.0},
                "flow_temporal_encoder": {
                    "attention_layers": 2,
                    "attention_heads": 1,
                    "attention_dim": 16,
                    "attention_feedforward_dim": 16,
                    "dropout": 0.0,
                },
                "copula_temporal_encoder": {
                    "attention_layers": 2,
                    "attention_heads": 1,
                    "attention_dim": 16,
                    "attention_feedforward_dim": 16,
                    "dropout": 0.0,
                },
                "copula_decoder": {
                    "min_u": 0.05,
                    "max_u": 0.95,
                    "attentional_copula": {
                        "attention_heads": 3,
                        "attention_layers": 1,
                        "attention_dim": 8,
                        "mlp_layers": 2,
                        "mlp_dim": 48,
                        "resolution": 20,
                        "attention_mlp_class": "_simple_linear_projection",
                        "dropout": 0.0,
                        "activation_function": "relu",
                    },
                    "dsf_marginal": {
                        "mlp_layers": 2,
                        "mlp_dim": 48,
                        "flow_layers": 2,
                        "flow_hid_dim": 48,
                    },
                },
                "experiment_mode": "forecasting",
                "skip_copula": self.skip_copula, # Usually True for simplified mode
            }

    def _prepare_dataset(self, df: pd.DataFrame, freq: str = "M") -> ListDataset:
        """
        Converts a multivariate DataFrame with NaN (ragged edge) into GluonTS ListDataset.
        """
        # TACTiS expects (num_series, num_timesteps)
        target = df.values.T # Transpose to (C, T)
        
        # GluonTS ListDataset requires a start date and a target
        # We use the first index as start
        start_date = df.index[0]
        
        # Note: TACTiS handles missing values via the observed mask if provided,
        # but the estimator usually applies AddObservedValuesIndicator transformation.
        # We just pass the target with NaNs.
        
        return ListDataset(
            [{"start": start_date, "target": target}],
            freq=freq,
            one_dim_target=False
        )

    def fit_predict_fill(
        self,
        X_visible_m: pd.DataFrame,
        target_quarter_end: pd.Timestamp,
        cutoff_date: pd.Timestamp,
        dataset_type: str,
        target_quarter: str,
        vintage_label: str,
        exclude_cols: list[str] = ["gdp_target"],
        force_refit: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main entry point for filling.
        """
        if not TRACTIS_AVAILABLE:
            raise ImportError(
                "TACTiS dependencies (gluonts, pytorchts) not installed. Use 'pip install tactis[research]' or manual setup.")

        exclude_cols = [str(c) for c in exclude_cols]
        X_visible_m = X_visible_m.copy()
        X_visible_m.columns = X_visible_m.columns.astype(str)
        seed_suffix = f"_s{self.random_state}"
        panel_filename = f"X_filled_tactis2_{dataset_type}_{target_quarter}_{vintage_label}{seed_suffix}.parquet"
        panel_path = self.cache_dir / panel_filename
        
        if panel_path.exists() and not force_refit:
            logger.info(f"Loading cached TACTiS2 filled panel from {panel_path}")
            X_filled_m = pd.read_parquet(panel_path)
            # We don't return the audit DF if loading from cache for now (unless saved separately)
            return X_filled_m, pd.DataFrame()

        # 1. Identify series needing filling and group them by last_valid_date
        cols = [c for c in X_visible_m.columns if c not in exclude_cols]
        X_subset = X_visible_m[cols].copy()
        
        origin_groups = {} # pd.Timestamp -> list of column names
        for col in X_subset.columns:
            s = X_subset[col].dropna()
            if s.empty:
                continue
            last_date = s.index[-1]
            if last_date < target_quarter_end:
                if last_date not in origin_groups:
                    origin_groups[last_date] = []
                origin_groups[last_date].append(col)
        
        # 2. Preparation
        # 2. Preparation
        X_filled_m = X_visible_m.copy()

        # Parquet / sklearn safety: all column names must be strings
        X_filled_m.columns = X_filled_m.columns.astype(str)
        X_subset.columns = X_subset.columns.astype(str)
        if target_quarter_end > X_filled_m.index.max():
            future_idx = pd.date_range(
                start=X_filled_m.index.max() + pd.offsets.MonthEnd(1),
                end=target_quarter_end,
                freq='M'
            )
            X_filled_m = X_filled_m.reindex(X_filled_m.index.union(future_idx))

        audit_rows = []
        total_forecasted_values = 0
        total_ffill_fallback_values = 0
        group_failures = 0
        processed_origins = []
        
        t0 = time.time()
        
        # 3. Process each origin group
        sorted_origins = sorted(origin_groups.keys())
        logger.info(f"TACTiS2: Found {len(sorted_origins)} unique origin groups.")
        
        for origin_date in sorted_origins:
            group_cols = origin_groups[origin_date]
            processed_origins.append(str(origin_date.date()))
            
            logger.info(f"Processing origin group: {origin_date.date()} ({len(group_cols)} series)")
            
            # Truncate full multivariate panel to this origin
            X_train_input = X_subset[X_subset.index <= origin_date]
            
            # Prediction length to reach target_quarter_end
            actual_prediction_length = len(pd.date_range(start=origin_date + pd.offsets.MonthEnd(1), 
                                                        end=target_quarter_end, freq='M'))
            
            if actual_prediction_length <= 0:
                continue

            # Safety check: TACTiS2/GluonTS transformation can fail if prediction_length 
            # is too large for the available history, or if it's too long for a nowcasting model.
            # We also skip TACTiS2 for horizons > 4 years as it's not designed for long-term stagnant filling.
            min_req_history = 24 # We want at least 2 years of history to train a meaningful factor
            if actual_prediction_length > 48 or len(X_train_input) < (actual_prediction_length + min_req_history):
                logger.warning(f"Origin {origin_date}: Horizon {actual_prediction_length} or history {len(X_train_input)} unsuitable for TACTiS2. Falling back to ffill.")
                for col_name in group_cols:
                    s_to_fill = X_filled_m.loc[:target_quarter_end, col_name]
                    last_v = s_to_fill.dropna().iloc[-1] if not s_to_fill.dropna().empty else 0
                    fill_idx = s_to_fill[s_to_fill.isna()].index
                    X_filled_m.loc[fill_idx, col_name] = last_v
                    total_ffill_fallback_values += len(fill_idx)
                    
                    for f_date in fill_idx:
                        audit_rows.append({
                            "dataset_type": dataset_type,
                            "target_quarter": target_quarter,
                            "vintage_label": vintage_label,
                            "cutoff_date": cutoff_date,
                            "origin_date": origin_date,
                            "series_name": col_name,
                            "forecast_date": f_date,
                            "forecast_step": -1,
                            "forecast_value": last_v,
                            "fill_method": "ffill",
                            "model_backend": "fallback_short_history",
                            "status": "tactis2_skipped_ffill"
                        })
                continue

            # History length adjustment: ensure history_length + prediction_length <= total_length
            actual_history_length = min(self.context_length, len(X_train_input) - actual_prediction_length)
            actual_history_length = max(min_req_history, actual_history_length)
            
            try:
                # 3a. Initialize and train for this origin
                # We reuse parameters but need a fresh estimator for the specific prediction_length
                if self.author_config:
                    trainer = TACTISTrainer(
                        epochs_phase_1=self.epochs_phase_1,
                        epochs_phase_2=self.epochs_phase_2,
                        batch_size=self.batch_size,
                        training_num_batches_per_epoch=self.num_batches_per_epoch,
                        learning_rate=self.learning_rate,
                        weight_decay=self.weight_decay,
                        clip_gradient=self.clip_gradient,
                        device=self.device,
                        seed=self.random_state,
                        checkpoint_dir=str(self.checkpoint_dir),
                        skip_batch_size_search=True,
                    )
                else:
                    trainer = TACTISTrainer(
                        epochs=self.max_epochs,
                        batch_size=self.batch_size,
                        training_num_batches_per_epoch=self.num_batches_per_epoch,
                        learning_rate=self.learning_rate,
                        device=self.device,
                        seed=self.random_state,
                        checkpoint_dir=str(self.checkpoint_dir),
                        skip_batch_size_search=True,
                    )
                
                estimator = TACTiSEstimator(
                    model_parameters=self.model_parameters,
                    num_series=X_subset.shape[1],
                    history_length=actual_history_length,
                    prediction_length=actual_prediction_length,
                    freq="M",
                    trainer=trainer,
                    cdf_normalization=False,
                    num_parallel_samples=self.num_samples,
                )
                
                train_ds = self._prepare_dataset(X_train_input)
                trained_net = estimator.train(train_ds, train_ds, num_workers=0, prefetch_factor=None)
                
                transformation = estimator.create_transformation()
                predictor = estimator.create_predictor(
                    transformation=transformation,
                    trained_network=trained_net,
                    device=self.device,
                    history_length=actual_history_length,
                )
                
                forecasts = list(predictor.predict(train_ds))
                forecast = forecasts[0]
                
                # 3b. Robust shape handling
                samples = forecast.samples
                fill_values_raw = np.mean(samples, axis=0)
                
                if fill_values_raw.shape == (X_subset.shape[1], actual_prediction_length):
                    fill_values = fill_values_raw.T # (H, C)
                elif fill_values_raw.shape == (actual_prediction_length, X_subset.shape[1]):
                    fill_values = fill_values_raw # (H, C)
                else:
                    raise ValueError(f"Unexpected shape: {fill_values_raw.shape}")
                
                # 3c. Map to X_filled_m for group columns ONLY
                forecast_dates = pd.date_range(start=origin_date + pd.offsets.MonthEnd(1), 
                                              periods=actual_prediction_length, freq='M')
                
                for col_name in group_cols:
                    col_idx = X_subset.columns.get_loc(col_name)
                    for i, f_date in enumerate(forecast_dates):
                        val = fill_values[i, col_idx]
                        X_filled_m.loc[f_date, col_name] = val
                        total_forecasted_values += 1
                        
                        audit_rows.append({
                            "dataset_type": dataset_type,
                            "target_quarter": target_quarter,
                            "vintage_label": vintage_label,
                            "cutoff_date": cutoff_date,
                            "origin_date": origin_date,
                            "series_name": col_name,
                            "forecast_date": f_date,
                            "forecast_step": i + 1,
                            "forecast_value": val,
                            "fill_method": "tactis2",
                            "model_backend": "ServiceNow/TACTiS",
                            "status": "tactis2_ok"
                        })
            
            except Exception as e:
                logger.error(f"TACTiS2 failed for origin {origin_date}: {e}")
                group_failures += 1
                # Fallback to ffill for this group
                for col_name in group_cols:
                    s_to_fill = X_filled_m.loc[:target_quarter_end, col_name]
                    last_v = s_to_fill.dropna().iloc[-1] if not s_to_fill.dropna().empty else 0
                    fill_idx = s_to_fill[s_to_fill.isna()].index
                    X_filled_m.loc[fill_idx, col_name] = last_v
                    total_ffill_fallback_values += len(fill_idx)
                    
                    for f_date in fill_idx:
                        audit_rows.append({
                            "dataset_type": dataset_type,
                            "target_quarter": target_quarter,
                            "vintage_label": vintage_label,
                            "cutoff_date": cutoff_date,
                            "origin_date": origin_date,
                            "series_name": col_name,
                            "forecast_date": f_date,
                            "forecast_step": -1,
                            "forecast_value": last_v,
                            "fill_method": "ffill",
                            "model_backend": "fallback",
                            "status": "tactis2_group_failed_ffill"
                        })

        # 4. Final safety ffill for non-group gaps (if any)
        X_filled_m = X_filled_m.ffill()

        # 5. Save and Return
        X_filled_m.columns = X_filled_m.columns.astype(str)

        X_filled_m.to_parquet(panel_path)
        panel_experiment_path = self.panel_out_dir / panel_filename
        X_filled_m.to_parquet(panel_experiment_path)

        audit_df = pd.DataFrame(audit_rows)
        if not audit_df.empty:
            audit_df.columns = audit_df.columns.astype(str)

            audit_filename = f"tactis2_filled_values_{dataset_type}_s{self.random_state}.csv"
            audit_path = self.output_dir / audit_filename
            header = not audit_path.exists()
            audit_df.to_csv(audit_path, mode='a', index=False, header=header)
            
        diagnostics = {
            "tactis2_origin_groups": len(sorted_origins),
            "tactis2_values_forecasted": total_forecasted_values,
            "tactis2_values_ffill_fallback": total_ffill_fallback_values,
            "tactis2_group_failures": group_failures,
            "tactis2_origin_dates_used": "|".join(processed_origins),
            "tactis2_runtime_sec": round(time.time() - t0, 2),
            "tactis2_author_config": self.author_config,
            "tactis2_max_epochs": self.max_epochs,
            "tactis2_epochs_phase_1": self.epochs_phase_1,
            "tactis2_epochs_phase_2": self.epochs_phase_2,
            "tactis2_batch_size": self.batch_size,
            "tactis2_num_batches_per_epoch": self.num_batches_per_epoch,
            "tactis2_learning_rate": self.learning_rate,
            "tactis2_weight_decay": self.weight_decay,
            "tactis2_maximum_learning_rate": self.maximum_learning_rate,
            "tactis2_clip_gradient": self.clip_gradient,
            "tactis2_bagging_size": self.bagging_size,
            "tactis2_skip_copula": self.skip_copula if not self.author_config else False,
            "tactis2_num_samples": self.num_samples
        }
            
        return X_filled_m, diagnostics
