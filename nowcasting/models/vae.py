"""
vae.py — Variational Autoencoder Nowcast Model
================================================

Architecture
------------
1. VAE Encoder: compress the high-dimensional panel (PCA-reduced) into a
   low-dimensional Gaussian latent space (mu, log_var).
2. Reparameterisation trick: z = mu + eps * std.
3. VAE Decoder: reconstruct input from z (trained jointly via ELBO loss).
4. Regression head: after training the VAE, freeze the encoder and fit a
   linear ElasticNetCV regression from the latent mean mu to the GDP target.

At inference: encode X_test → mu → ElasticNetCV → GDP prediction.

Why VAE for nowcasting?
-----------------------
- Learns a smooth, disentangled latent representation of the macroeconomic panel.
- Robust to noise and missing values (filled before encoding).
- The latent space acts as a learned nonlinear dimensionality reduction,
  complementing PCA (linear) used by other models.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from nowcasting.models.base import BaseNowcastModel

logger = logging.getLogger("nowcast_vae")


# ---------------------------------------------------------------------------
# VAE Network
# ---------------------------------------------------------------------------

class _VAENet(nn.Module):
    """
    Simple fully-connected VAE.

    Parameters
    ----------
    input_dim  : number of input features (e.g. 10 PCA components)
    hidden_dim : width of hidden layers
    latent_dim : dimensionality of the latent space z
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 8):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.fc_mu      = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var


def _vae_loss(x_recon: torch.Tensor, x: torch.Tensor,
              mu: torch.Tensor, log_var: torch.Tensor,
              beta: float = 1.0) -> torch.Tensor:
    """beta-VAE ELBO loss = reconstruction MSE + beta * KL divergence."""
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss


# ---------------------------------------------------------------------------
# VAE Nowcast Model
# ---------------------------------------------------------------------------

class VAENowcast(BaseNowcastModel):
    """
    Variational Autoencoder Nowcasting model.

    Phase 1 — Unsupervised VAE training on X_train:
        Learns a latent encoding of the macroeconomic feature panel.

    Phase 2 — Supervised regression on latent representations:
        Encodes X_train → mu, then fits ElasticNetCV(mu, y_train).

    Prediction:
        Encode X_test → mu → ElasticNetCV → y_pred.

    Parameters
    ----------
    target_col  : str
    horizon     : int
    latent_dim  : int — dimensionality of the latent space (default 8)
    hidden_dim  : int — hidden layer width (default 64)
    epochs      : int — VAE training epochs (default 30)
    lr          : float — Adam learning rate
    beta        : float — KL weight in beta-VAE loss (1.0 = standard VAE)
    device      : 'cpu' | 'cuda' | None (auto-detect)
    """

    def __init__(
        self,
        target_col: str,
        horizon: int = 1,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        epochs: int = 30,
        lr: float = 1e-3,
        beta: float = 1.0,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(target_col, horizon, **kwargs)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs     = epochs
        self.lr         = lr
        self.beta       = beta
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler  = StandardScaler()
        self.reg     = ElasticNetCV(cv=5, random_state=123, max_iter=3000)
        self.vae_net: Optional[_VAENet] = None

    # ------------------------------------------------------------------
    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32).to(self.device)

    # ------------------------------------------------------------------
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "VAENowcast":
        if X_train.empty:
            logger.warning("VAENowcast.fit: empty X_train.")
            return self

        # --- Preprocess ---
        X_np = self.scaler.fit_transform(X_train.fillna(0))
        input_dim = X_np.shape[1]

        # --- Phase 1: train VAE ---
        logger.info(
            f"VAENowcast: training VAE | input_dim={input_dim}, "
            f"latent={self.latent_dim}, epochs={self.epochs}, device={self.device}"
        )
        self.vae_net = _VAENet(input_dim, self.hidden_dim, self.latent_dim).to(self.device)
        optimiser = torch.optim.Adam(self.vae_net.parameters(), lr=self.lr)

        X_t = self._to_tensor(X_np)
        self.vae_net.train()
        for epoch in range(self.epochs):
            optimiser.zero_grad()
            x_recon, mu, log_var = self.vae_net(X_t)
            loss = _vae_loss(x_recon, X_t, mu, log_var, self.beta)
            loss.backward()
            optimiser.step()
            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                logger.info(f"  VAE epoch {epoch+1}/{self.epochs}  loss={loss.item():.4f}")

        # --- Phase 2: encode → regress ---
        self.vae_net.eval()
        with torch.no_grad():
            mu_enc, _ = self.vae_net.encode(X_t)
            latent_np = mu_enc.cpu().numpy()   # (T, latent_dim)

        # Align with target (drop rows with NaN y)
        y_arr = y_train.values
        valid  = ~np.isnan(y_arr)
        if valid.sum() < 3:
            logger.warning("VAENowcast: too few non-NaN target observations for regression.")
            return self

        self.reg.fit(latent_np[valid], y_arr[valid])
        self.is_fitted = True
        logger.info("VAENowcast: regression fitted on latent representations.")
        return self

    # ------------------------------------------------------------------
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if not self.is_fitted or self.vae_net is None:
            raise ValueError("VAENowcast not fitted.")

        X_np = self.scaler.transform(X_test.fillna(0))
        X_t  = self._to_tensor(X_np)

        self.vae_net.eval()
        with torch.no_grad():
            mu_enc, _ = self.vae_net.encode(X_t)
            latent_np = mu_enc.cpu().numpy()

        preds = self.reg.predict(latent_np)
        return pd.Series(preds, index=X_test.index, name=f"{self.target_col}_pred")
