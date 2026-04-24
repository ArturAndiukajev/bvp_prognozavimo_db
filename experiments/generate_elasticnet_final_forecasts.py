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
# Naudojame ElasticNetNowcast klasę iš jūsų ml_regression.py failo
from nowcasting.models.ml_regression import ElasticNetNowcast
from nowcasting.evaluation.backtester import RollingBacktester

# Nustatome logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enet_final_forecast")

def main():
    # 2. KONFIGŪRACIJOS IR DUOMENŲ KELIAI
    # Naudojame jūsų geriausios konfigūracijos failą
    config_name = 'optuna_best_config_common_final_nowcast_dataset_s123.json'
    config_path = base_dir / 'data' / 'forecasts' / config_name
    dataset_name = "common_final_nowcast_dataset.parquet"
    data_dir = base_dir / 'data' / 'processed'
    
    if not config_path.exists():
        logger.error(f"Konfigūracijos failas nerastas: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Kraunami Elastic Net parametrai iš: {config_path.name}")

    # 3. PARAMETRŲ GAVIMAS TIESIAI IŠ KONFIGŪRACIJOS (be hard-coding)
    # Kadangi JSON'e nėra target_col, naudojame gdp_target kaip numatytąjį
    target_col = config.get('target_col', 'gdp_target')
    
    # Modelio nustatymai (l1_ratio, max_iter, fill_strategy)
    model_params = config.get('model_params', "{}")
    if isinstance(model_params, str):
        model_params = json.loads(model_params)
        
    # Kintamųjų atrankos (selector) nustatymai (fast_screen, top_n)
    selector_method = config.get('selector_method', 'none')
    selector_params = config.get('selector_params', "{}")
    if isinstance(selector_params, str):
        selector_params = json.loads(selector_params)

    # 4. DUOMENŲ UŽKROVIMAS
    X, y = load_cf_panel(data_dir, target_col, panel_arg=dataset_name)

    # 5. MODELIO INICIJAVIMAS
    # Naudojame ElasticNetNowcast, kuris tiesiogiai palaiko Elastic Net ir Selector 
    model = ElasticNetNowcast(
        model_type=config.get('model', 'elasticnet'),
        model_params=model_params,
        selector_method=selector_method,
        selector_params=selector_params,
        target_col=target_col,
        fill_strategy=model_params.get('fill_strategy', 'mean'),
        seed=config.get('seed', 123)
    )

    # 6. ILGOS ISTORIJOS GENERAVIMAS (Backtest)
    # Nustatome initial_train_periods=40, kad gautume ilgą seką (nuo ~2010 m.)
    backtester = RollingBacktester(
        initial_train_periods=40, 
        step_size=1,               
        window_type="expanding"
    )

    logger.info(f"Pradedamas Elastic Net prognozavimas (Atranka: {selector_method})...")
    
    # Tikriname teisingą metodo pavadinimą (run, backtest arba evaluate)
    try:
        forecasts = backtester.run(model, X, y)
    except AttributeError:
        try:
            forecasts = backtester.backtest(model, X, y)
        except AttributeError:
            logger.info("Bandomas .evaluate() metodas")
            forecasts = backtester.evaluate(model, X, y)

    # 7. IŠSAUGOME REZULTATUS
    output_name = f"FINAL_LONG_FORECAST_ENET.csv"
    output_path = base_dir / 'data' / 'forecasts' / output_name
    forecasts.to_csv(output_path, index=False)
    
    logger.info(f"Sugeneruota {len(forecasts)} prognozių.")
    logger.info(f"Rezultatai išsaugoti: {output_path}")

if __name__ == "__main__":
    main()