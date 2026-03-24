"""
selectors.py - Sklearn-compatible feature selectors and compressors
===================================================================

Provides standardized transformers for grid-searching feature engineering 
before passing data to models (e.g., Dynamic Factor Models).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("nowcast_selectors")


class IdentitySelector(BaseEstimator, TransformerMixin):
    """Passes data through unchanged ('none' strategy)."""
    def __init__(self):
        self.feature_names_in_ = None

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = X.columns
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class FastScreeningFilter(BaseEstimator, TransformerMixin):
    """
    Highly scalable first-pass filter for very large arrays.
    Filters by:
      1. Minimum valid observations ratio (completeness_threshold)
      2. Minimum variance (variance_threshold)
      3. Top K features by maximum absolute correlation with the target
         over a specified lag window (max_lag).
    """
    def __init__(
        self,
        completeness_threshold: float = 0.5,
        variance_threshold: float = 1e-6,
        top_k: int = 100,
        max_lag: int = 3,
    ):
        self.completeness_threshold = completeness_threshold
        self.variance_threshold = variance_threshold
        self.top_k = top_k
        self.max_lag = max_lag
        self.selected_cols_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if X.empty:
            self.selected_cols_ = []
            self.is_fitted_ = True
            return self

        # 1. Completeness Filter
        valid_ratio = X.notna().mean()
        candidate_cols = valid_ratio[valid_ratio >= self.completeness_threshold].index

        if len(candidate_cols) == 0:
            logger.warning("FastScreeningFilter: no feature met the completeness threshold.")
            self.selected_cols_ = []
            self.is_fitted_ = True
            return self

        X_sub = X[candidate_cols]

        # 2. Variance Filter
        variances = X_sub.var(skipna=True)
        candidate_cols = variances[variances >= self.variance_threshold].index

        if len(candidate_cols) == 0:
            logger.warning("FastScreeningFilter: no feature met the variance threshold.")
            self.selected_cols_ = []
            self.is_fitted_ = True
            return self

        X_sub = X_sub[candidate_cols]

        # 3. Lag-Aware Absolute Correlation
        # Create shifted targets inside the train sample
        y_lags = pd.DataFrame({f"lag_{i}": y.shift(i) for i in range(self.max_lag + 1)})
        y_lags = y_lags.dropna(how="all")
        
        common_idx = X_sub.index.intersection(y_lags.index)
        X_align = X_sub.loc[common_idx]
        y_align = y_lags.loc[common_idx]

        corrs_max = {}
        for col in X_align.columns:
            valid = X_align[col].notna()
            n_valid = valid.sum()
            
            if n_valid > 5:
                x_val = X_align.loc[valid, col].values
                y_val = y_align.loc[valid].values
                
                best_c = 0.0
                for lag_idx in range(self.max_lag + 1):
                    y_l = y_val[:, lag_idx]
                    v_mask = ~np.isnan(y_l)
                    if v_mask.sum() > 5:
                        # Compute safely across remaining valid overlapping items
                        corr_matrix = np.corrcoef(x_val[v_mask], y_l[v_mask])
                        if corr_matrix.shape == (2, 2):
                            c = corr_matrix[0, 1]
                            if not np.isnan(c) and abs(c) > best_c:
                                best_c = abs(c)
                corrs_max[col] = best_c
            else:
                corrs_max[col] = 0.0

        corr_s = pd.Series(corrs_max).sort_values(ascending=False)
        self.selected_cols_ = corr_s.head(self.top_k).index.tolist()
        
        logger.info(f"FastScreen: condensed {X.shape[1]} -> {len(candidate_cols)} -> {len(self.selected_cols_)} features.")
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_cols_:
            return X.iloc[:, :0]  # Return empty keeping index
        return X[self.selected_cols_].copy()


class VarianceFilter(BaseEstimator, TransformerMixin):
    """Removes near-zero variance features."""
    def __init__(self, threshold: float = 1e-6):
        self.threshold = threshold
        self.selector = VarianceThreshold(threshold=self.threshold)
        self.feature_names_in_ = None
        self.selected_cols_ = None

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = X.columns
        # Temporarily fill NA with median for variance calculation
        X_filled = X.fillna(X.median())
        self.selector.fit(X_filled)
        
        support = self.selector.get_support()
        self.selected_cols_ = X.columns[support]
        logger.info(f"VarianceFilter: kept {len(self.selected_cols_)} / {len(X.columns)} features.")
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_cols_].copy()


class CorrTopNSelector(BaseEstimator, TransformerMixin):
    """Selects top N features by absolute Pearson correlation with target."""
    def __init__(self, top_n: int = 50):
        self.top_n = top_n
        self.selected_cols_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Align X and y to drop NaNs
        common_idx = X.index.intersection(y.dropna().index)
        X_align = X.loc[common_idx]
        y_align = y.loc[common_idx]

        corrs = {}
        for col in X_align.columns:
            # Drop pair-wise NaNs
            valid = X_align[col].notna()
            if valid.sum() > 5:
                # Absolute correlation
                corr = np.abs(np.corrcoef(X_align.loc[valid, col], y_align.loc[valid])[0, 1])
                corrs[col] = corr if not np.isnan(corr) else 0.0
            else:
                corrs[col] = 0.0

        corr_s = pd.Series(corrs).sort_values(ascending=False)
        # Select top N valid
        self.selected_cols_ = corr_s.head(self.top_n).index.tolist()
        logger.info(f"CorrTopN: selected top {len(self.selected_cols_)} features.")
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_cols_:
            return X
        return X[self.selected_cols_].copy()


class LassoSelector(BaseEstimator, TransformerMixin):
    """Uses Lasso regression coefficients magnitude to select features."""
    def __init__(self, alpha: float = 0.1, max_features: Optional[int] = None):
        self.alpha = alpha
        self.max_features = max_features
        self.selected_cols_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Align and fill NA for linear model fitting
        common_idx = X.index.intersection(y.dropna().index)
        X_align = X.loc[common_idx].fillna(X.median()).fillna(0)
        y_align = y.loc[common_idx]

        # Standardize features before Lasso
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_align)

        lasso = Lasso(alpha=self.alpha, max_iter=2000)
        lasso.fit(X_scaled, y_align)

        coefs = pd.Series(np.abs(lasso.coef_), index=X.columns)
        nonzero = coefs[coefs > 0].sort_values(ascending=False)
        
        if self.max_features and len(nonzero) > self.max_features:
            selected = nonzero.head(self.max_features).index.tolist()
        else:
            selected = nonzero.index.tolist()
            
        # Fallback if alpha is too high and everything zeroes out
        if not selected:
            logger.warning(f"Lasso(alpha={self.alpha}) zeroed all features. Falling back to top 5.")
            selected = coefs.sort_values(ascending=False).head(5).index.tolist()

        self.selected_cols_ = selected
        logger.info(f"Lasso: selected {len(self.selected_cols_)} features.")
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_cols_].copy()


class ElasticNetSelector(BaseEstimator, TransformerMixin):
    """Uses ElasticNet regression coefficients magnitude to select features."""
    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5, max_features: Optional[int] = None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_features = max_features
        self.selected_cols_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        common_idx = X.index.intersection(y.dropna().index)
        X_align = X.loc[common_idx].fillna(X.median()).fillna(0)
        y_align = y.loc[common_idx]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_align)

        en = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=2000)
        en.fit(X_scaled, y_align)

        coefs = pd.Series(np.abs(en.coef_), index=X.columns)
        nonzero = coefs[coefs > 0].sort_values(ascending=False)
        
        if self.max_features and len(nonzero) > self.max_features:
            selected = nonzero.head(self.max_features).index.tolist()
        else:
            selected = nonzero.index.tolist()
            
        if not selected:
            logger.warning(f"ElasticNet(alpha={self.alpha}) zeroed all features. Falling back to top 5.")
            selected = coefs.sort_values(ascending=False).head(5).index.tolist()

        self.selected_cols_ = selected
        logger.info(f"ElasticNet: selected {len(self.selected_cols_)} features.")
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_cols_].copy()


class PCACompressor(BaseEstimator, TransformerMixin):
    """Reduces dataset completely using PCA."""
    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.pca = None
        self.scaler = None

    def fit(self, X: pd.DataFrame, y=None):
        X_filled = X.fillna(X.median()).fillna(0)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_filled)

        n_comp = min(self.n_components, X_scaled.shape[0], X_scaled.shape[1])
        self.pca = PCA(n_components=n_comp)
        self.pca.fit(X_scaled)
        logger.info(f"PCA: fitted {n_comp} components (req: {self.n_components}).")
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_filled = X.fillna(X.median()).fillna(0)
        X_scaled = self.scaler.transform(X_filled)
        comp = self.pca.transform(X_scaled)
        
        cols = [f"PC_{i+1}" for i in range(comp.shape[1])]
        return pd.DataFrame(comp, index=X.index, columns=cols)


class FactorAnalysisCompressor(BaseEstimator, TransformerMixin):
    """Reduces dataset completely using Sklearn FactorAnalysis."""
    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.fa = None
        self.scaler = None

    def fit(self, X: pd.DataFrame, y=None):
        X_filled = X.fillna(X.median()).fillna(0)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_filled)

        n_comp = min(self.n_components, X_scaled.shape[0], X_scaled.shape[1])
        # FA is iterative and can take time, limit max_iter for speed
        self.fa = FactorAnalysis(n_components=n_comp, max_iter=100)
        self.fa.fit(X_scaled)
        logger.info(f"FactorAnalysis: fitted {n_comp} factors.")
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_filled = X.fillna(X.median()).fillna(0)
        X_scaled = self.scaler.transform(X_filled)
        comp = self.fa.transform(X_scaled)
        
        cols = [f"FA_{i+1}" for i in range(comp.shape[1])]
        return pd.DataFrame(comp, index=X.index, columns=cols)


class AutoencoderCompressor(BaseEstimator, TransformerMixin):
    """
    Experimental PyTorch Autoencoder compressor.
    Fails gracefully if PyTorch is not available.
    """
    def __init__(self, latent_dim: int = 5, hidden_dim: int = 32, epochs: int = 20, batch_size: int = 16):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = None
        self.device = None
        self._is_valid = False

        try:
            import torch
            import torch.nn as nn
            self._is_valid = True
            
            class SimpleAE(nn.Module):
                def __init__(self, input_dim, hidden_dim, latent_dim):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, latent_dim)
                    )
                def forward(self, x):
                    return self.encoder(x)
                    
            self.model_class = SimpleAE
        except ImportError:
            logger.warning("PyTorch not installed. AutoencoderCompressor will act as an Identity pass-through.")

    def fit(self, X: pd.DataFrame, y=None):
        if not self._is_valid:
            return self

        import torch
        import torch.nn as nn
        import torch.optim as optim

        X_filled = X.fillna(X.median()).fillna(0)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_filled)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = X_scaled.shape[1]
        
        # Real AE requires decoder for training
        class TrainableAE(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim)
                )
            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z)

        ae = TrainableAE(input_dim, self.hidden_dim, self.latent_dim).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(ae.parameters(), lr=1e-3)

        tensor_X = torch.FloatTensor(X_scaled).to(self.device)
        dataset = torch.utils.data.TensorDataset(tensor_X)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        ae.train()
        for epoch in range(self.epochs):
            for batch in loader:
                x_batch = batch[0]
                optimizer.zero_grad()
                recon = ae(x_batch)
                loss = criterion(recon, x_batch)
                loss.backward()
                optimizer.step()

        # Save only encoder for fast inference
        self.model = ae.encoder
        self.model.eval()
        
        logger.info(f"Autoencoder: trained {self.latent_dim} dimension latent space.")
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_valid or self.model is None:
            return X
            
        import torch
        X_filled = X.fillna(X.median()).fillna(0)
        X_scaled = self.scaler.transform(X_filled)
        
        with torch.no_grad():
            tensor_X = torch.FloatTensor(X_scaled).to(self.device)
            latent = self.model(tensor_X).cpu().numpy()
            
        cols = [f"AE_{i+1}" for i in range(latent.shape[1])]
        return pd.DataFrame(latent, index=X.index, columns=cols)
