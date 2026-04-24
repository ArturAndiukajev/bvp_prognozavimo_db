import pandas as pd
import json
import sys
import logging
from pathlib import Path

# 1. APLINKOS PARUOŠIMAS
current_dir = Path(__file__).resolve().parent
base_dir = current_dir.parent
sys.path.append(str(base_dir))

from nowcasting.utils.data_loader import load_cf_panel
from nowcasting.models.midas import MIDASNowcast
from nowcasting.evaluation.backtester import RollingBacktester

# Nustatome logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("final_forecast")

def main():
    # 2. ĮKELIAME GERIAUSIUS PARAMETRUS
    config_path = base_dir / 'data' / 'forecasts' / 'optuna_best_config_midas_mixed_final_nowcast_dataset_s123.json'
    
    if not config_path.exists():
        logger.error(f"Konfigūracijos failas nerastas: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Kraunami geriausi parametrai iš: {config_path.name}")

    # 3. DUOMENŲ NUSTATYMAS
    target_col = config.get('target_col', 'gdp_target')
    dataset_name = "mixed_final_nowcast_dataset.parquet"
    data_dir = base_dir / 'data' / 'processed'
    
    logger.info(f"Tikslinis kintamasis: {target_col} | Duomenys: {dataset_name}")

    # 4. UŽKRAUKITE DUOMENIS
    X, y = load_cf_panel(data_dir, target_col, panel_arg=dataset_name)

    # 5. INICIJUOKITE MODELĮ
    reg_params = config.get('regression_params', "{}")
    if isinstance(reg_params, str):
        reg_params = json.loads(reg_params)

    model = MIDASNowcast(
        target_col=target_col,
        n_lags=config.get('n_lags', 6), 
        regression_model=config.get('regression_model', 'elasticnet'),
        regression_kwargs=reg_params,
        fill_strategy=config.get('fill_strategy', 'mean')
    )

    # 6. VYKDOME ILGĄJĮ TESTAVIMĄ (BACKTEST)
    backtester = RollingBacktester(
        initial_train_periods=40, 
        step_size=1,               
        window_type="expanding"
    )

    logger.info("Pradedamas ilgos istorijos generavimas")
    
    # --- PAKEITIMAS ČIA: Naudojame .run() vietoj .test() ---
    try:
        # Jei .run() neveiks, bandykite .backtest()
        forecasts = backtester.run(model, X, y)
    except AttributeError:
        logger.warning(".run() nerastas, bandoma naudoti .backtest()")
        forecasts = backtester.backtest(model, X, y)

    # 7. IŠSAUGOME REZULTATUS
    output_name = f"FINAL_LONG_FORECAST_MIDAS.csv"
    output_path = base_dir / 'data' / 'forecasts' / output_name
    forecasts.to_csv(output_path, index=False)
    
    logger.info(f"Sugeneruota {len(forecasts)} prognozių.")
    logger.info(f"Rezultatai išsaugoti: {output_path}")

if __name__ == "__main__":
    main()