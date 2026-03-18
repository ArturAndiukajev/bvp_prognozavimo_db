import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nowcasting.models.base import BaseNowcastModel

class PCARegressionNowcast(BaseNowcastModel):
    """
    Principal Component Regression.
    Extracts k principal components from the panel and regress the target on them.
    """
    def __init__(self, target_col: str, horizon: int = 1, n_components: int = 5, **kwargs):
        super().__init__(target_col, horizon, **kwargs)
        self.n_components = n_components
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("regressor", LinearRegression())
        ])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        valid_idx = y_train.notna()
        # PCA requires complete data, so we must fill NaNs. 
        # For simplicity, filling with 0 (which is the mean after StandardScaler) 
        # natively built into Pipeline logic if we did it sequentially, but here we fill before scaling.
        # A more advanced approach uses EM-PCA, but standard PCA + zero-fill is normal baseline.
        X_train_clean = X_train.loc[valid_idx].fillna(X_train.mean().fillna(0))
        y_train_clean = y_train.loc[valid_idx]
        
        self.model.fit(X_train_clean, y_train_clean)
        self.X_mean_ = X_train.mean().fillna(0) # Store for inference
        self.is_fitted = True
        return self

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
            
        X_test_clean = X_test.fillna(self.X_mean_)
        preds = self.model.predict(X_test_clean)
        return pd.Series(preds, index=X_test.index, name=f"{self.target_col}_pred")
