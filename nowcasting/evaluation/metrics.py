import pandas as pd
import numpy as np

def compute_metrics(eval_df: pd.DataFrame, model_name: str = "Model") -> pd.DataFrame:
    """
    Given a dataframe with Actual and Predicted columns from the backtester,
    returns standard macro forecasting metrics.
    """
    clean_df = eval_df.dropna()
    
    if clean_df.empty:
        return pd.DataFrame({
            "Model": [model_name],
            "RMSE": [np.nan],
            "MAE": [np.nan],
            "MAPE": [np.nan]
        })

    y_true = clean_df["Actual"]
    y_pred = clean_df["Predicted"]
    err = y_true - y_pred
    
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    
    mape = np.mean(np.abs(err / np.where(y_true == 0, 1e-8, y_true))) * 100
    
    metrics = {
        "Model": [model_name],
        "RMSE": [rmse],
        "MAE": [mae],
        "MAPE": [mape]
    }
    
    return pd.DataFrame(metrics)
