import pandas as pd
import numpy as np
import json
import logging
import sys
import glob
import os
import re
from pathlib import Path
from sqlalchemy import create_engine, text
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

# 1. APLINKOS PARUOŠIMAS
current_dir = Path(__file__).resolve().parent
base_dir = current_dir.parent
sys.path.append(str(base_dir))

from nowcasting.utils.data_loader import load_cf_panel
from nowcasting.models.midas import MIDASNowcast

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("impact_diagnose")

DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"

# ==========================================
# PAVADINIMŲ ATITAIKYMO FUNKCIJA (IŠMANI)
# ==========================================
def map_variable_names(impact_df):
    """
    Ištraukia bazinius ID iš modelio rezultatų ir užklausia TIK juos iš DB,
    kad išvengtų RAM atminties perpildymo (OOM Killed klaidos).
    """
    if impact_df.empty:
        return impact_df
        
    try:
        # 1. Bazinio ID ištraukimas (be lagų, jei tai MIDAS)
        impact_df['Base_ID'] = impact_df['Variable'].astype(str).str.split('_').str[0]
        
        # 2. Gauname unikalius ID, kurių mums reikia
        unique_ids = impact_df['Base_ID'].unique().tolist()
        
        if not unique_ids:
            return impact_df
            
        # 3. Formuojame tikslinę SQL užklausą
        id_list_str = "', '".join(unique_ids)
        query = text(f"SELECT id, key, name FROM series WHERE id::text IN ('{id_list_str}')")
        
        logger.info(f"Traukiami {len(unique_ids)} kintamųjų metaduomenys iš DB...")
        
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            df_db = pd.read_sql(query, conn)
            
        if df_db.empty:
            logger.warning("DB negrąžino jokių atitikmenų šiems ID.")
            impact_df['Series_Key'] = "Nerasta_DB"
            impact_df['Name'] = "Nerasta_DB"
            return impact_df
            
        # 4. Sukuriame žodynus iš mažos DB lentelės
        df_db['str_id'] = df_db['id'].astype(str)
        key_dict = dict(zip(df_db['str_id'], df_db['key']))
        name_dict = dict(zip(df_db['str_id'], df_db['name']))
        
        # 5. Priskiriame duomenis
        impact_df['Raw_Series_Key'] = impact_df['Base_ID'].map(key_dict).fillna("Nežinomas_Key")
        impact_df['Name'] = impact_df['Base_ID'].map(name_dict).fillna("Nežinomas kintamasis")
        
        # --- TEKSTO VALYMAS ---
        # Nukerpame viską, kas yra po '|' ženklo ir pašaliname nereikalingus tarpus
        impact_df['Name'] = impact_df['Name'].astype(str).str.split('|').str[0].str.strip()
        # Taip pat išvalome galimus skliaustelius
        impact_df['Name'] = impact_df['Name'].str.replace(r'\(.*?\)', '', regex=True).str.strip()
        
        # 6. Tvarkome MIDAS lagus (pvz. _lag1_t3)
        impact_df['Suffix'] = impact_df['Variable'].astype(str).apply(
            lambda x: f"_{x.split('_', 1)[1]}" if '_' in x else ""
        )
        
        # Formuojame galutinį Series_Key
        impact_df['Series_Key'] = impact_df['Raw_Series_Key'] + impact_df['Suffix']

        # 7. Grąžiname tvarkingą struktūrą
        cols = ['Variable', 'Series_Key', 'Name', 'Coefficient', 'Abs_Impact']
        return impact_df[cols]
            
    except Exception as e:
        logger.error(f"Klaida priskiriant pavadinimus: {e}")
        if 'Series_Key' not in impact_df.columns:
            impact_df['Series_Key'] = "Klaida"
            impact_df['Name'] = "Klaida"
        return impact_df

# ==========================================
# DIAGNOSTIKOS FUNKCIJOS
# ==========================================
def get_non_zero_coeffs(model, feature_names, model_name):
    impact_df = pd.DataFrame({
        'Variable': feature_names,
        'Coefficient': model.coef_ if hasattr(model, 'coef_') else model.model.coef_
    })
    non_zero = impact_df[impact_df['Coefficient'] != 0].copy()
    non_zero['Abs_Impact'] = non_zero['Coefficient'].abs()
    return non_zero.sort_values('Abs_Impact', ascending=False)


def run_elastic_net_diagnostics(base_dir, data_dir, output_dir):
    logger.info("\n" + "="*30 + "\n1. ELASTIC NET DIAGNOSTIKA\n" + "="*30)
    
    config_path = base_dir / 'data' / 'forecasts' / 'optuna_trials_common_final_nowcast_dataset_s123.csv'
    if not config_path.exists():
        logger.error("Elastic Net konfigūracija nerasta!")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    X, y = load_cf_panel(data_dir, config.get('target_col', 'gdp_target'), panel_arg="common_final_nowcast_dataset.parquet")
    
    y_clean = y.dropna()
    X_clean = X.loc[y_clean.index].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    reg_params = config.get('regression_params', {})
    if isinstance(reg_params, str): reg_params = json.loads(reg_params)

    alphas_to_test = [reg_params.get('alpha', 1.0), 0.1, 0.01, 0.001, 0.0001]
    best_impact = pd.DataFrame()
    final_alpha = alphas_to_test[0]

    for a in alphas_to_test:
        test_params = reg_params.copy()
        test_params['alpha'] = a
        model = ElasticNet(**test_params, random_state=123)
        model.fit(X_scaled, y_clean)
        impact = get_non_zero_coeffs(model, X.columns, "ElasticNet")
        
        if len(impact) > 0:
            best_impact = impact
            final_alpha = a
            break
    
    if len(best_impact) == 0:
        logger.error("Visi koeficientai yra 0.")
    else:
        best_impact = map_variable_names(best_impact)
        
        logger.info(f"Naudojama Alpha: {final_alpha} | Rasta kintamųjų: {len(best_impact)}")
        pd.set_option('display.max_colwidth', 100)
        print(best_impact[['Variable', 'Series_Key', 'Name', 'Coefficient']].head(15).to_string(index=False))
        
        out_file = output_dir / "elasticnet_impact_results.csv"
        best_impact.to_csv(out_file, index=False)
        logger.info(f"Išsaugota: {out_file}")


def run_midas_diagnostics(base_dir, data_dir, output_dir):
    logger.info("\n" + "="*30 + "\n2. MIDAS DIAGNOSTIKA\n" + "="*30)
    
    config_path = base_dir / 'data' / 'forecasts' / 'optuna_best_config_midas_mixed_final_nowcast_dataset_s123.json'
    if not config_path.exists():
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    X, y = load_cf_panel(data_dir, config.get('target_col', 'gdp_target'), panel_arg="mixed_final_nowcast_dataset.parquet")
    
    reg_params = config.get('regression_params', "{}")
    if isinstance(reg_params, str): reg_params = json.loads(reg_params)

    model = MIDASNowcast(
        target_col=config.get('target_col', 'gdp_target'),
        n_lags=config.get('n_lags', 6),
        regression_model=config.get('regression_model', 'elasticnet'),
        regression_kwargs=reg_params,
        fill_strategy=config.get('fill_strategy', 'mean')
    )

    model.fit(X, y)
    
    if model.is_fitted:
        impact = get_non_zero_coeffs(model, model._feature_cols, "MIDAS")
        
        impact = map_variable_names(impact)
        
        logger.info(f"Rasta kintamųjų: {len(impact)}")
        pd.set_option('display.max_colwidth', 100)
        print(impact[['Variable', 'Series_Key', 'Name', 'Coefficient']].head(15).to_string(index=False))
        
        out_file = output_dir / "midas_impact_results.csv"
        impact.to_csv(out_file, index=False)
        logger.info(f"Išsaugota: {out_file}")
    else:
        logger.error("MIDAS modelio nepavyko apmokyti.")

if __name__ == '__main__':
    data_dir = base_dir / 'data' / 'processed'
    run_elastic_net_diagnostics(base_dir, data_dir, current_dir)
    run_midas_diagnostics(base_dir, data_dir, current_dir)