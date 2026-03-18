import abc
import pandas as pd
from typing import Dict, Any

class BaseNowcastModel(abc.ABC):
    """
    Standard interface for all forecasting and nowcasting models.
    Every model must support fit() and predict().
    """
    def __init__(self, target_col: str, horizon: int = 1, **kwargs):
        self.target_col = target_col
        self.horizon = horizon
        self.params = kwargs
        self.is_fitted = False

    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Base validation: checks for empty frames, full NaN columns, and 
        index alignment. Called inside fit/predict pipelines.
        """
        import numpy as np
        
        if X is None or X.empty:
            raise ValueError(f"{self.__class__.__name__}: Input features X form an empty DataFrame.")
            
        # Optional basic feature bounds checks
        if X.isna().all().any():
            bad_cols = X.columns[X.isna().all()].tolist()
            raise ValueError(f"{self.__class__.__name__}: Input X contains completely NaN columns: {bad_cols[:3]}")
            
        if y is not None:
            if y.empty:
                raise ValueError(f"{self.__class__.__name__}: Target y is empty.")
            if not y.index.intersection(X.index).empty is False and len(y.index.intersection(X.index)) == 0:
                 raise ValueError(f"{self.__class__.__name__}: Predictor X and target y indices do not overlap.")

    @abc.abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'BaseNowcastModel':
        """
        Fits the model. Models should internally handle alignment if ragged, 
        or rely on the backtester to supply pre-cleaned panels.
        """
        pass

    @abc.abstractmethod
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Predicts over the test horizon.
        """
        pass
        
    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """
        Calculates basic metrics.
        """
        import numpy as np
        if len(y_true) != len(y_pred):
            # Align first if necessary, here we assume already aligned.
            common = y_true.index.intersection(y_pred.index)
            y_true = y_true.loc[common]
            y_pred = y_pred.loc[common]

        err = y_true - y_pred
        rmse = np.sqrt(np.mean(err**2))
        mae = np.mean(np.abs(err))
        
        # Avoid division by zero
        mape = np.mean(np.abs(err / np.where(y_true == 0, 1e-8, y_true))) * 100

        return {
            "RMSE": float(rmse),
            "MAE": float(mae),
            "MAPE": float(mape)
        }
