"""
tactis_wrapper.py  —  TACTiS Forecasting Wrapper (Dense API)
================================================================

Key improvements over the original:
1. **Plain Forecasting (No Data Hallucination)**: 
   This wrapper acts as a pure unconditional forecaster, drawing on the memory of the sequence seen in `train_tail`.
   Because the raw dense PyTorch TACTiS API fundamentally lacks robust NaN masking or jagged multi-variate indexing without resorting to the GluonTS framework, attempting to condition on `X_test` via 0.0 filling causes severe data hallucinations. This wrapper intentionally side-steps that by using strict conditional temporal alignment without introducing falsified test-window observations.
   
2. **Two-Stage Curriculum Training**: Implements Stage 1 flow unrolling before initializing and updating Stage 2 attention copulas per TACTiS official methodology.

3. **Probabilistic Outputs (`predict_quantiles`)**: Enables retention of full distribution variance without eagerly collapsing to deterministic estimates.

2. **Seed / reproducibility**: `seed` parameter threads through numpy, torch,
   cuda, and the DataLoader generator.

3. **Configurable architecture**: all key TACTiS hyperparameters exposed as
   constructor params with sensible defaults matching the original.

4. **Feature selection**: optional `feature_selector` attribute for
   dimensionality reduction before feeding series to TACTiS (fit on train data
   only, no leakage).
"""

from __future__ import annotations

import logging
import random
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from nowcasting.models.base import BaseNowcastModel

logger = logging.getLogger("nowcast_tactis")


# ---------------------------------------------------------------------------
# Reproducibility helper
# ---------------------------------------------------------------------------

def set_torch_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducible TACTiS runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class TACTiSNowcastWrapper(BaseNowcastModel):
    """
    Improved TACTiS wrapper focusing on plain forecasting.
    Due to raw dense PyTorch API constraints, this does NOT condition on X_test new variables natively to avoid Data Hallucination.

    Parameters
    ----------
    target_col : str
        Name of the target series.
    horizon : int
        Forecast horizon (low-frequency periods ahead).
    history_length : int
        Context window (number of time steps fed to TACTiS as history).
    epochs : int
        Training epochs.
    batch_size : int
        Mini-batch size during training.
    lr : float
        Adam learning rate.
    num_samples : int
        Number of Monte-Carlo samples drawn at inference; median is returned.
    device : str or None
        Torch device string ('cpu', 'cuda'). Auto-detected if None.
    seed : int
        Global random seed for reproducibility.
    skip_copula : bool
        If True, use flow-only (Stage 1) model — faster but less expressive.
        For nowcasting experimentation this is often a good starting point.

    TACTiS architecture params (all keyword-only):
        flow_series_embedding_dim, copula_series_embedding_dim,
        flow_input_encoder_layers, copula_input_encoder_layers,
        flow_attention_layers, flow_attention_heads,
        flow_attention_dim, flow_attention_feedforward_dim,
        copula_attention_layers, copula_attention_heads,
        copula_attention_dim, copula_attention_feedforward_dim,
        copula_mlp_layers, copula_mlp_dim,
        flow_layers, flow_hid_dim
    """

    def __init__(
        self,
        target_col: str,
        horizon: int = 1,
        # Training params
        history_length: int = 24,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        num_samples: int = 100,
        device: Optional[str] = None,
        seed: int = 123,
        # Architecture params
        skip_copula: bool = False,
        flow_series_embedding_dim: int = 5,
        copula_series_embedding_dim: int = 5,
        flow_input_encoder_layers: int = 2,
        copula_input_encoder_layers: int = 2,
        flow_attention_layers: int = 2,
        flow_attention_heads: int = 1,
        flow_attention_dim: int = 16,
        flow_attention_feedforward_dim: int = 16,
        copula_attention_layers: int = 2,
        copula_attention_heads: int = 1,
        copula_attention_dim: int = 16,
        copula_attention_feedforward_dim: int = 16,
        copula_mlp_layers: int = 2,
        copula_mlp_dim: int = 48,
        flow_layers: int = 2,
        flow_hid_dim: int = 48,
        **kwargs,
    ):
        super().__init__(target_col, horizon, **kwargs)

        # Training
        self.history_length = history_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_samples = num_samples
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prediction_type = "pure_forecast"

        # Architecture
        self.skip_copula = skip_copula
        self._arch = dict(
            flow_series_embedding_dim=flow_series_embedding_dim,
            copula_series_embedding_dim=copula_series_embedding_dim,
            flow_input_encoder_layers=flow_input_encoder_layers,
            copula_input_encoder_layers=copula_input_encoder_layers,
            flow_attention_layers=flow_attention_layers,
            flow_attention_heads=flow_attention_heads,
            flow_attention_dim=flow_attention_dim,
            flow_attention_feedforward_dim=flow_attention_feedforward_dim,
            copula_attention_layers=copula_attention_layers,
            copula_attention_heads=copula_attention_heads,
            copula_attention_dim=copula_attention_dim,
            copula_attention_feedforward_dim=copula_attention_feedforward_dim,
            copula_mlp_layers=copula_mlp_layers,
            copula_mlp_dim=copula_mlp_dim,
            flow_layers=flow_layers,
            flow_hid_dim=flow_hid_dim,
        )

        self.scaler = StandardScaler()
        self.tactis_model = None
        self.is_fitted = False

        # Will be set during fit()
        self.columns_: List[str] = []
        self._target_idx: int = -1
        self._last_train_scaled: Optional[np.ndarray] = None  # shape [T, n_series]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_tactis_model(self, n_series: int):
        from tactis.model.tactis import TACTiS

        a = self._arch
        model_parameters = {
            "num_series": n_series,
            "flow_series_embedding_dim": a["flow_series_embedding_dim"],
            "copula_series_embedding_dim": a["copula_series_embedding_dim"],
            "flow_input_encoder_layers": a["flow_input_encoder_layers"],
            "copula_input_encoder_layers": a["copula_input_encoder_layers"],
            "bagging_size": None,
            "input_encoding_normalization": True,
            "data_normalization": "standardization",
            "loss_normalization": "both",
            "positional_encoding": {"dropout": 0.0},
            "flow_temporal_encoder": {
                "attention_layers": a["flow_attention_layers"],
                "attention_heads": a["flow_attention_heads"],
                "attention_dim": a["flow_attention_dim"],
                "attention_feedforward_dim": a["flow_attention_feedforward_dim"],
                "dropout": 0.0,
            },
            "copula_temporal_encoder": {
                "attention_layers": a["copula_attention_layers"],
                "attention_heads": a["copula_attention_heads"],
                "attention_dim": a["copula_attention_dim"],
                "attention_feedforward_dim": a["copula_attention_feedforward_dim"],
                "dropout": 0.0,
            },
            "copula_decoder": {
                "min_u": 0.05,
                "max_u": 0.95,
                "attentional_copula": {
                    "attention_heads": 3,
                    "attention_layers": 1,
                    "attention_dim": 16,
                    "mlp_layers": a["copula_mlp_layers"],
                    "mlp_dim": a["copula_mlp_dim"],
                    "resolution": 20,
                    "attention_mlp_class": "_simple_linear_projection",
                    "dropout": 0.0,
                    "activation_function": "relu",
                },
                "dsf_marginal": {
                    "mlp_layers": 2,
                    "mlp_dim": a["copula_mlp_dim"],
                    "flow_layers": a["flow_layers"],
                    "flow_hid_dim": a["flow_hid_dim"],
                },
            },
            "experiment_mode": "forecasting",
            "skip_copula": self.skip_copula,
        }
        return TACTiS(**model_parameters)

    def _create_tactis_windows(self, scaled_arr: np.ndarray):
        """
        Build (hist_time, hist_val, pred_time, pred_val) windows
        from a [T, n_series] scaled array.
        """
        n_total, n_series = scaled_arr.shape
        hl = self.history_length
        hz = self.horizon

        windows_hist, windows_pred = [], []
        for i in range(n_total - hl - hz + 1):
            windows_hist.append(scaled_arr[i : i + hl].T)         # [series, hl]
            windows_pred.append(scaled_arr[i + hl : i + hl + hz].T)  # [series, hz]

        if not windows_hist:
            return None, None, None, None

        t_index = np.arange(n_total)
        hist_times, pred_times = [], []
        for i in range(n_total - hl - hz + 1):
            ht = np.tile(t_index[i : i + hl], (n_series, 1))
            pt = np.tile(t_index[i + hl : i + hl + hz], (n_series, 1))
            hist_times.append(ht)
            pred_times.append(pt)

        hist_val  = torch.tensor(np.array(windows_hist), dtype=torch.float32)
        pred_val  = torch.tensor(np.array(windows_pred), dtype=torch.float32)
        hist_time = torch.tensor(np.array(hist_times),   dtype=torch.float32)
        pred_time = torch.tensor(np.array(pred_times),   dtype=torch.float32)
        return hist_time, hist_val, pred_time, pred_val

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "TACTiSNowcastWrapper":
        self._fit_counter = getattr(self, "_fit_counter", 0) + 1
        set_torch_seed(self.seed)
        logger.info(
            f"TACTiS fit step {self._fit_counter} | series={X_train.shape[1]+1}  "
            f"epochs={self.epochs}  hl={self.history_length}  "
            f"skip_copula={self.skip_copula}  seed={self.seed}"
        )

        # Use ffill().bfill() instead of fillna(0.0) to prevent aggressive 0.0 bias during StandardScaler fit
        full_train = pd.concat([X_train, y_train.rename(self.target_col)], axis=1).ffill().bfill().fillna(0.0)
        self.columns_ = list(full_train.columns)
        self._target_idx = self.columns_.index(self.target_col)
        n_series = len(self.columns_)

        scaled_vals = self.scaler.fit_transform(full_train.values)
        self._last_train_scaled = scaled_vals  # store for predict()

        hist_time, hist_val, pred_time, pred_val = self._create_tactis_windows(scaled_vals)

        if hist_val is None:
            logger.warning("TACTiS.fit: not enough data to create training windows. Skipping.")
            return self

        dataset = TensorDataset(hist_time, hist_val, pred_time, pred_val)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        self.tactis_model = self._init_tactis_model(n_series).to(self.device)

        # --- PHASE 1: Train Flow Marginals ---
        self.tactis_model.train()
        self.tactis_model.set_stage(1)
        optimizer_stage1 = torch.optim.Adam(self.tactis_model.parameters(), lr=self.lr)
        epochs_stage1 = self.epochs // 2
        
        for epoch in range(epochs_stage1):
            total_loss = 0.0
            for ht, hv, pt, pv in loader:
                ht = ht.to(self.device)
                hv = hv.to(self.device)
                pt = pt.to(self.device)
                pv = pv.to(self.device)
                optimizer_stage1.zero_grad()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    marginal_logdet, _ = self.tactis_model.loss(
                        hist_time=ht, hist_value=hv, pred_time=pt, pred_value=pv
                    )
                loss = -marginal_logdet.mean()
                loss.backward()
                optimizer_stage1.step()
                total_loss += loss.item()
            logger.info(f"  [Stage 1] epoch {epoch+1}/{epochs_stage1}  loss={total_loss:.4f}")

        # --- PHASE 2: Train Attentional Copula ---
        if not self.skip_copula:
            self.tactis_model.set_stage(2)
            self.tactis_model.initialize_stage2()
            self.tactis_model.to(self.device)  # Re-send components to device after init
            optimizer_stage2 = torch.optim.Adam(self.tactis_model.parameters(), lr=self.lr)
            epochs_stage2 = self.epochs - epochs_stage1
            
            for epoch in range(epochs_stage2):
                total_loss = 0.0
                for ht, hv, pt, pv in loader:
                    ht = ht.to(self.device)
                    hv = hv.to(self.device)
                    pt = pt.to(self.device)
                    pv = pv.to(self.device)
                    optimizer_stage2.zero_grad()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _, copula_loss = self.tactis_model.loss(
                            hist_time=ht, hist_value=hv, pred_time=pt, pred_value=pv
                        )
                    loss = copula_loss.mean()
                    loss.backward()
                    optimizer_stage2.step()
                    total_loss += loss.item()
                logger.info(f"  [Stage 2] epoch {epoch+1}/{epochs_stage2}  loss={total_loss:.4f}")

        self.is_fitted = True
        return self

    def _predict_tensors(self, X_test: pd.DataFrame) -> torch.Tensor:
        """Internal helper for standardizing prediction tensor creation and inference."""
        if not self.is_fitted:
            raise RuntimeError("TACTiSNowcastWrapper is not fitted yet.")

        set_torch_seed(self.seed)

        steps = len(X_test)
        n_series = len(self.columns_)

        # --- Build context ---
        # Take the exact training tail and use ONLY that as the history context.
        # This removes the "0 target masking data hallucination" and properly matches TACTiS dense API expectations.
        train_tail = self._last_train_scaled[-self.history_length :]  # [hl, n_series]
        
        self._predict_counter = getattr(self, "_predict_counter", 0) + 1
        logger.info(
            f"TACTiS predict step {self._predict_counter} | test_steps (horizon)={steps} | "
            f"pure forecast (no 0-padding test context)"
        )

        hv = torch.tensor(train_tail.T, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, series, hl]
        ht = (
            torch.arange(self.history_length, dtype=torch.float32)
            .unsqueeze(0).unsqueeze(0)
            .expand(1, n_series, -1)
            .to(self.device)
        )
        
        # Proper temporal alignment: prediction time explicitly corresponds to the steps directly after train tail
        pt = (
            torch.arange(self.history_length, self.history_length + steps, dtype=torch.float32)
            .unsqueeze(0).unsqueeze(0)
            .expand(1, n_series, -1)
            .to(self.device)
        )

        # --- Sample ---
        self.tactis_model.eval()
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                samples = self.tactis_model.sample(
                    num_samples=self.num_samples,
                    hist_time=ht,
                    hist_value=hv,
                    pred_time=pt,
                )
        return samples

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Deterministic median nowcast prediction.
        """
        samples = self._predict_tensors(X_test)
        steps = len(X_test)
        
        # samples: [1, n_series, steps, num_samples]
        target_samples = samples[0, self._target_idx, -steps :, :]  # [steps, num_samples]
        target_scaled_median = target_samples.median(dim=-1).values.cpu().numpy()  # [steps]

        col_mean  = self.scaler.mean_[self._target_idx]
        col_std   = self.scaler.scale_[self._target_idx]
        target_preds = target_scaled_median * col_std + col_mean

        return pd.Series(
            target_preds,
            index=X_test.index,
            name=f"{self.target_col}_pred",
        )

    def predict_quantiles(self, X_test: pd.DataFrame, quantiles: List[float] = [0.1, 0.5, 0.9]) -> pd.DataFrame:
        """
        Returns predictions at specified quantiles to preserve probabilistic estimates.
        """
        samples = self._predict_tensors(X_test)
        steps = len(X_test)
        
        target_samples = samples[0, self._target_idx, -steps :, :]  # [steps, num_samples]
        
        col_mean  = self.scaler.mean_[self._target_idx]
        col_std   = self.scaler.scale_[self._target_idx]
        
        df_dict = {}
        for q in quantiles:
            quantile_tensor = torch.quantile(target_samples, q, dim=-1).cpu().numpy()
            target_preds = quantile_tensor * col_std + col_mean
            df_dict[f"q_{int(q*100)}"] = target_preds
            
        return pd.DataFrame(df_dict, index=X_test.index)

    def forecast(self, history: pd.DataFrame, steps: int = 1) -> pd.DataFrame:
        """
        Direct multi-step forecast from an arbitrary history frame.
        Uses only the provided history (no X_test conditioning).
        Kept for backward compatibility.
        """
        if not self.is_fitted:
            raise RuntimeError("TACTiSNowcastWrapper is not fitted yet.")

        set_torch_seed(self.seed)

        n_series = len(self.columns_)
        # Use ffill().bfill() instead of fillna(0.0) to maintain true scale of the history tail
        hist_df   = history.reindex(columns=self.columns_).ffill().bfill().fillna(0.0)
        hist_tail = hist_df.iloc[-self.history_length :]
        scaled    = self.scaler.transform(hist_tail.values)

        hv = torch.tensor(scaled.T, dtype=torch.float32).unsqueeze(0).to(self.device)
        ht = (
            torch.arange(len(hist_tail), dtype=torch.float32)
            .unsqueeze(0).unsqueeze(0)
            .expand(1, n_series, -1)
            .to(self.device)
        )
        pt = (
            torch.arange(len(hist_tail), len(hist_tail) + steps, dtype=torch.float32)
            .unsqueeze(0).unsqueeze(0)
            .expand(1, n_series, -1)
            .to(self.device)
        )

        self.tactis_model.eval()
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                samples = self.tactis_model.sample(
                    num_samples=self.num_samples,
                    hist_time=ht,
                    hist_value=hv,
                    pred_time=pt,
                )

        preds_scaled = samples[:, :, -steps:, :].median(dim=-1).values.squeeze(0).cpu().numpy().T
        dummy = pd.DataFrame(preds_scaled, columns=self.columns_)
        preds = self.scaler.inverse_transform(dummy)
        return pd.DataFrame(preds, columns=self.columns_)