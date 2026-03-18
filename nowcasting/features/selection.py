import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger("nowcast_selection")

class MultiStageFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Performs dimensionality reduction using Variance Thresholding, 
    Correlation culling (highly correlated features), or PCA.
    Must ONLY be fit on the train set.
    """
    def __init__(self, method: str = "pca", n_components: int = 5, variance_threshold: float = 1e-5):
        self.method = method
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        
        self.var_thresh_ = VarianceThreshold(threshold=variance_threshold)
        self.pca_ = PCA(n_components=n_components)
        self.selected_features_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fits the selection mechanism (PCA or univariate filters).
        y is only strictly required if using Mutual Information.
        """
        logger.info(f"Fitting selector ({self.method}) on shape {X.shape}...")
        
        # 1. Always drop zero/low variance globally
        X_clean = X.fillna(0) # For learning relations, fillna is ok. PCA can't take NaNs.
        self.var_thresh_.fit(X_clean)
        X_clean = X_clean.loc[:, self.var_thresh_.get_support()]

        if self.method == "pca":
            n_features_out = X_clean.shape[1]
            actual_n_components = min(self.n_components, n_features_out, X_clean.shape[0])
            self.pca_.set_params(n_components=actual_n_components)
            self.pca_.fit(X_clean)
        elif self.method == "mutual_info" and y is not None:
            # We must drop target NAs for MI
            valid_idx = y.notna()
            mi_scores = mutual_info_regression(X_clean.loc[valid_idx], y.loc[valid_idx])
            mi_series = pd.Series(mi_scores, index=X_clean.columns)
            self.selected_features_ = mi_series.nlargest(self.n_components).index.tolist()
        else:
            # Fallback for pure variance / correlation (not fully implemented)
            self.selected_features_ = X_clean.columns.tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Transforming features using {self.method}...")
        
        X_clean = X.fillna(0) # For application
        
        try:
            X_clean = X_clean.loc[:, self.var_thresh_.get_support()]
            
            if self.method == "pca":
                pca_features = self.pca_.transform(X_clean)
                return pd.DataFrame(
                    pca_features, 
                    index=X.index, 
                    columns=[f"PC_{i+1}" for i in range(pca_features.shape[1])]
                )
            elif self.method == "mutual_info":
                return X_clean[self.selected_features_]
            else:
                return X_clean
        except Exception as e:
            logger.error(f"Failed feature transform: {e}. Returning original.")
            return X
