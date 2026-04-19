"""
DFM eksperimento paleidimo failas su Grid Search, Optuna ir feature selection.

Šis skriptas leidžia vykdyti eksperimentus su Dynamic Factor Model (DFM),
testuojant skirtingas hiperparametrų kombinacijas ir kintamųjų atrankos metodus.

Pagrindinės funkcijos:
Grid Search per DFM hiperparametrus ir feature selection metodus
Optuna optimizacija
Staged (dviejų etapų) paieška greitesniam filtravimui
Backtesting (rolling / expanding window)
Parallelizavimas (joblib) spartesniam vykdymui
Automatinis rezultatų išsaugojimas (CSV / JSON)
Galimybė atnaujinti (resume) nutrauktus eksperimentus

Palaikomi režimai:
grid      – pilnas kombinacijų perrinkimas
staged    – greitas dviejų etapų filtravimas
optuna    – pažangi hiperparametrų optimizacija
evaluate  – vienos konfigūracijos įvertinimas

Rezultatai:
Modelių palyginimas pagal RMSE / MAE
Geriausių konfigūracijų išsaugojimas
Prognozių eksportas į CSV

Pavyzdys:
python scripts/run_dfm_experiment.py
--selectors none,pca,lasso,elasticnet
--dfm-k-factors 1,2,3
--pca-components 3,5
--lasso-alphas 0.05,0.1
--n-jobs 4
--search-mode quick
"""


import argparse
import itertools
import json
import logging
import os
import time
import random
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import optuna
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# Force single-thread linear algebra to prevent collision in parallel workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Ensure project root is importable when run as a script directly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nowcasting.main import build_arg_parser as build_main_parser
from nowcasting.evaluation.checkpoint import prune_grid, run_chunks, build_run_signature
from nowcasting.utils.data_loader import load_cf_panel, load_mf_panels, _DEFAULT_DATA_DIR
from nowcasting.evaluation.backtester import RollingBacktester
from nowcasting.evaluation.metrics import compute_metrics
from nowcasting.models.dfm import DynamicFactorNowcast

from nowcasting.features.selectors import (
    IdentitySelector, VarianceFilter, PCACompressor, FactorAnalysisCompressor,
    LassoSelector, ElasticNetSelector, CorrTopNSelector, AutoencoderCompressor, FastScreeningFilter
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("dfm_experiment")


class WindowStandardScaler(BaseEstimator, TransformerMixin):
    """
    StandardScaler that operates on a single train window, preserving NaNs and
    using medians for temporary imputation during scaling.
    """
    def __init__(self):
        self.scaler_ = StandardScaler()
        self.fill_values_ = None
        self.columns_ = None

    def fit(self, X, y=None):
        self.columns_ = X.columns
        # Compute median for imputation before scaling (only for numeric columns)
        self.fill_values_ = X.median(numeric_only=True)
        # Impute with median, then with 0 if medians are NaN (for constant columns)
        X_filled = X.fillna(self.fill_values_).fillna(0.0)
        self.scaler_.fit(X_filled)
        return self

    def transform(self, X):
        if self.fill_values_ is None:
            raise ValueError("Scaler must be fitted before transform.")
        
        # Save NaN mask to restore later
        mask = X.isna()
        
        # Impute with stored train median
        X_filled = X.fillna(self.fill_values_).fillna(0.0)
        
        # Scale
        X_scaled = self.scaler_.transform(X_filled)
        
        # Reconstruct DataFrame
        X_out = pd.DataFrame(X_scaled, index=X.index, columns=self.columns_)
        
        # Restore NaNs
        return X_out.where(~mask, np.nan)


def build_experiment_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="DFM Feature Selection Experiment Runner")
    
    # Data arguments 
    ap.add_argument("--data-dir", type=str, default=_DEFAULT_DATA_DIR)
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--input-panel", type=str, default=None)
    ap.add_argument("--mixed-frequency", action="store_true", help="Run in mixed-frequency mode")
    ap.add_argument("--quarterly-cols", type=str, default="", help="Comma-separated quarterly columns for MF DFM")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for grid search sampling")
    
    # Grid Search Spaces (Comma-separated lists)
    ap.add_argument("--selectors", type=str, default="none,fast_screen,pca,lasso,elasticnet,corr_top_n",
                    help="Comma-separated selectors to test: none,fast_screen,pca,lasso,elasticnet,corr_top_n,factor_analysis,autoencoder")
    ap.add_argument("--selectors-fast-only", type=str, default="fast_screen,corr_top_n", help="Optional subset of selectors for stage 1")
    
    ap.add_argument("--dfm-k-factors", type=str, default="1,2,3,5,8")
    ap.add_argument("--dfm-factor-orders", type=str, default="1,2,3")
    
    ap.add_argument("--pca-components", type=str, default="3,5,10,20")
    ap.add_argument("--factor-analysis-components", type=str, default="3,5")
    ap.add_argument("--lasso-alphas", type=str, default="0.01,0.1")
    ap.add_argument("--elasticnet-alphas", type=str, default="0.01,0.1")
    ap.add_argument("--elasticnet-l1-ratios", type=str, default="0.5")
    ap.add_argument("--corr-top-n", type=str, default="10,20,50,100,200")
    ap.add_argument("--ae-latent-dims", type=str, default="3,5")
    ap.add_argument("--fast-screen-top-k", type=str, default="50,100,250,500")

    ap.add_argument("--train-windows", type=str, default="120", 
                    help="Initial train window sizes (e.g., 80,120)")
    ap.add_argument("--step-sizes", type=str, default="1", 
                    help="Backtest step sizes (e.g., 1,3)")
    ap.add_argument("--window-type", type=str, default="expanding,rolling", help="expanding or rolling or both")
    ap.add_argument("--rolling-window-size", type=str, default="120", help="Rolling window size, relevant only if window-type includes rolling")

    ap.add_argument("--search-mode", type=str, choices=["quick", "full"], default="full",
                    help="Quick mode reduces maxiter and uses fewer backtest steps.")
    ap.add_argument("--search-last-n-steps", type=int, default=0,
                    help="Only run the backtest for the last N steps to save time.")
    ap.add_argument("--search-max-configs", type=int, default=0,
                    help="Limit the number of random configurations tested.")
    ap.add_argument("--search-strategy", type=str, default="full", choices=["full", "staged"],
                    help="Use 'staged' for a fast 1-pass coarse search to filter options.")
    ap.add_argument("--search-top-k", type=int, default=5,
                    help="Top configs to promote to final pass in staged search.")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Number of parallel workers for joblib.")

    # ---------- Optuna ----------
    ap.add_argument("--engine", choices=["optuna", "grid", "staged", "evaluate"], default=None,
                    help="Search engine to use: optuna, grid, staged, or evaluate (single run from config)")
    ap.add_argument("--config-path", type=str, default=None, help="Path to a JSON configuration file for '--engine evaluate'")
    ap.add_argument("--n-trials", type=int, default=50, help="Number of trials for Optuna engine")
    ap.add_argument("--study-storage", type=str, default=None, 
                    help="Optuna storage URI. Defaults to sqlite in results directory.")
    ap.add_argument("--study-name", type=str, default=None, help="Optuna study name for persistence and resuming.")
    ap.add_argument("--save-every-n-trials", type=int, default=5, help="Save summary CSV every N trials")
    
    ap.add_argument("--resume", action="store_true", help="Resume from an interrupted search using existing results CSV.")
    ap.add_argument("--checkpoint-chunk-size", type=int, default=10, help="Save to CSV after completing this many configurations.")
    ap.add_argument("--scale-x", action="store_true", help="Scale X per train window before feature selection/model fit")
    ap.add_argument("--out-file", type=str, default="data/forecasts/dfm_experiment_results.csv")

    return ap


def parse_list(val_str: str, dtype=int):
    return [dtype(v.strip()) for v in val_str.split(",") if v.strip()]

def build_selector(method: str, params: dict):
    if method == "none":
        return IdentitySelector()
    elif method == "pca":
        return PCACompressor(n_components=params.get("n_components", 5))
    elif method == "factor_analysis":
        return FactorAnalysisCompressor(n_components=params.get("n_components", 5))
    elif method == "lasso":
        return LassoSelector(alpha=params.get("alpha", 0.1))
    elif method == "elasticnet":
        return ElasticNetSelector(alpha=params.get("alpha", 0.1), l1_ratio=params.get("l1_ratio", 0.5))
    elif method == "corr_top_n":
        return CorrTopNSelector(top_n=params.get("top_n", 20))
    elif method == "autoencoder":
        return AutoencoderCompressor(latent_dim=params.get("latent_dim", 5), epochs=10)
    elif method == "fast_screen":
        return FastScreeningFilter(top_k=params.get("top_k", params.get("top_n", 50)))
    else:
        raise ValueError(f"Unknown selector method: {method}")


def run_single_experiment(
    config: Dict[str, Any],
    X_panel: pd.DataFrame, 
    y_target: pd.Series,
    search_mode: str,
    search_last_n_steps: int,
    scale_x: bool = False
) -> dict:
    """Runs a single combination grid configuration and returns metrics."""
    res = {**config}
    res["status"] = "running"
    res["scale_x"] = scale_x
    
    try:
        t0 = time.time()
        
        # 1. Backtester initialization
        init_train = config["train_window"]
        step_sz = config["step_size"]
        
        if search_last_n_steps > 0:
            # Shift the initial_train so it only runs the last N steps.
            required_start_index = len(y_target) - search_last_n_steps
            init_train = max(init_train, required_start_index)

        bt = RollingBacktester(
            initial_train_periods=init_train, 
            step_size=step_sz,
            window_type=config.get("window_type", "expanding"),
            eval_mode="mixed_frequency" if config.get("mixed_frequency") else "common_frequency",
            rolling_window_size=config.get("rolling_window_size")
        )

        # 2. Transformer initialization (scaling)
        scaler = WindowStandardScaler() if scale_x else None

        # 3. Selector Initialization
        # VarianceFilter is always implicitly added in front to drop constants safely
        from sklearn.pipeline import make_pipeline
        base_sel = build_selector(config["selector_method"], config["selector_params"])
        selector = make_pipeline(VarianceFilter(), base_sel)
        
        # 4. Model Initialization
        k = config["dfm_k_factors"]
        fo = config["dfm_factor_order"]
        mod_kws = {}
        if search_mode == "quick":
            mod_kws["maxiter"] = 50

        model = DynamicFactorNowcast(
            target_col=str(y_target.name), 
            k_factors=k, 
            factor_order=fo, 
            mixed_frequency=config.get("mixed_frequency", False),
            quarterly_cols=config.get("quarterly_cols", []),
            **mod_kws
        )
        
        # 4. Run loop
        # We temporarily disable internal statsmodels progress output inside parallel steps
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eval_df = bt.backtest(
                model=model,
                X_panel=X_panel,
                y_target=y_target,
                transformer=scaler,
                feature_selector=selector
            )
            
        runt = time.time() - t0
        res["runtime_sec"] = round(runt, 2)
        
        if eval_df.empty:
            res["status"] = "failed"
            res["error_message"] = "Backtester returned empty DataFrame"
            return res

        metrics = compute_metrics(eval_df, model_name=f"DFM_{k}f")
        res["rmse"] = metrics["RMSE"].iloc[0]
        res["mae"] = metrics["MAE"].iloc[0]
        res["status"] = "success"
        res["_eval_df"] = eval_df
        res["n_eval_points"] = len(eval_df)
        res["n_folds"]       = eval_df["Forecast_Origin"].nunique() if "Forecast_Origin" in eval_df.columns else len(eval_df)
        
        # Track feature counts seamlessly natively provided by the backtester updates
        res["n_raw_features"]    = int(eval_df["n_raw_features"].median()) if "n_raw_features" in eval_df else None
        res["n_trans_features"]  = int(eval_df["n_trans_features"].median()) if "n_trans_features" in eval_df else None
        res["n_sel_features"]    = int(eval_df["n_sel_features"].median()) if "n_sel_features" in eval_df else None
        res["n_model_used_vars"] = int(eval_df["n_model_used_features"].median()) if "n_model_used_features" in eval_df else None

        
    except Exception as e:
        logger.warning(f"Experiment failed: {config} -> {e}")
        res["status"] = "failed"
        res["error_message"] = str(e)
        
    return res


# ---------------------------------------------------------------------------
# Optuna Engine
# ---------------------------------------------------------------------------

class OptunaBestTracker:
    def __init__(self, out_dir: Path, suffix: str, initial_best: float = float("inf")):
        self.best_rmse = initial_best
        self.out_dir = out_dir
        self.suffix = suffix
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
def optuna_objective(
    trial: optuna.Trial, 
    X_panel: pd.DataFrame, 
    y_target: pd.Series, 
    args: argparse.Namespace, 
    tracker: OptunaBestTracker
) -> float:
    selector_name = trial.suggest_categorical("selector", ["none", "corr_top_n", "fast_screen", "lasso", "elasticnet", "pca", "factor_analysis", "autoencoder"])
    selector_params = {}
    
    if selector_name == "corr_top_n":
        selector_params["top_n"] = trial.suggest_int("corr_top_n_k", 5, 50)
    elif selector_name == "lasso":
        selector_params["alpha"] = trial.suggest_float("lasso_alpha", 1e-4, 1.0, log=True)
    elif selector_name == "elasticnet":
        selector_params["alpha"] = trial.suggest_float("elasticnet_alpha", 1e-4, 1.0, log=True)
        selector_params["l1_ratio"] = trial.suggest_float("elasticnet_l1", 0.1, 0.9)
    elif selector_name == "fast_screen":
        selector_params["top_n"] = trial.suggest_int("fast_screen_k", 20, 100)
    elif selector_name == "pca":
        selector_params["n_components"] = trial.suggest_int("pca_comp", 2, 20)
    elif selector_name == "factor_analysis":
        selector_params["n_components"] = trial.suggest_int("fa_comp", 2, 20)
    elif selector_name == "autoencoder":
        selector_params["latent_dim"] = trial.suggest_int("ae_dim", 2, 20)

    dfm_k_factors = trial.suggest_int("dfm_k_factors", 1, 4)
    dfm_factor_order = trial.suggest_int("dfm_factor_order", 1, 3)

    train_window = trial.suggest_categorical("train_window", [60, 80, 100, 120])
    step_size = trial.suggest_categorical("step_size", [1, 3])
    window_type = trial.suggest_categorical("window_type", ["expanding", "rolling"])
    
    config = {
        "selector_method": selector_name,
        "selector_params": selector_params,
        "dfm_k_factors": dfm_k_factors,
        "dfm_factor_order": dfm_factor_order,
        "train_window": train_window,
        "step_size": step_size,
        "window_type": window_type,
        "mixed_frequency": getattr(args, "mixed_frequency", False)
    }
    
    if window_type == "rolling":
        config["rolling_window_size"] = trial.suggest_categorical("rolling_window_size", [40, 60, 80])
        
    if getattr(args, "mixed_frequency", False):
        q_cols = getattr(args, "quarterly_cols", "")
        config["quarterly_cols"] = [c.strip() for c in q_cols.split(",") if c.strip()]
    
    res = run_single_experiment(
        config=config,
        X_panel=X_panel,
        y_target=y_target,
        search_mode=args.search_mode,
        search_last_n_steps=args.search_last_n_steps,
        scale_x=getattr(args, "scale_x", False)
    )
    
    for k, v in res.items():
        if k != "_eval_df":
            trial.set_user_attr(k, v)
            
    if res.get("status") != "success":
        raise optuna.TrialPruned(f"Run failed: {res.get('error_message', 'Unknown error')}")
        
    rmse = res.get("rmse")
    if rmse is None or pd.isna(rmse):
        raise optuna.TrialPruned("RMSE was NaN or None")
        
    if rmse < tracker.best_rmse:
        tracker.best_rmse = rmse
        if "_eval_df" in res and not res["_eval_df"].empty:
            preds_path = tracker.out_dir / f"dfm_optuna_best_preds_{tracker.suffix}.csv"
            res["_eval_df"].to_csv(preds_path, index=False)
            logger.info(f"New best RMSE ({rmse:.4f}) -> {preds_path}")
            
    return rmse


def run_optuna_engine(X_panel: pd.DataFrame, y_target: pd.Series, args: argparse.Namespace, out_dir: Path, suffix: str):
    logger.info("Starting OPTUNA engine.")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    study_name = args.study_name or f"dfm_{suffix}"
    storage_url = args.study_storage or f"sqlite:///{out_dir.absolute()}/dfm_optuna.db"
    
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
    )
    
    tracker = OptunaBestTracker(out_dir, suffix)
    
    if len(study.trials) > 0:
        logger.info(f"Resuming study '{study_name}' with {len(study.trials)} existing trials.")
        try:
            tracker.best_rmse = study.best_value
        except ValueError:
            pass

    out_csv = out_dir / f"optuna_trials_{suffix}.csv"
    
    def save_callback(study_cb: optuna.Study, trial_cb: optuna.trial.FrozenTrial):
        if trial_cb.number % args.save_every_n_trials == 0:
            df_trials = study_cb.trials_dataframe()
            df_trials.to_csv(out_csv, index=False)
            try:
                best_trial = study_cb.best_trial
                best_dict = best_trial.user_attrs.copy()
                best_dict["rmse"] = study_cb.best_value
                with open(out_dir / f"optuna_best_config_{suffix}.json", "w") as f:
                    json.dump(best_dict, f, indent=4, default=str)
            except ValueError:
                pass

    study.optimize(
        lambda t: optuna_objective(t, X_panel, y_target, args, tracker),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        catch=(Exception,),
        callbacks=[save_callback]
    )
    
    df_trials = study.trials_dataframe()
    df_trials.to_csv(out_csv, index=False)
    logger.info(f"Optuna trials saved to {out_csv}")
    
    try:
        best_trial = study.best_trial
        logger.info(f"Optuna Best RMSE: {best_trial.value:.4f}")
        logger.info(f"Optuna Best Params: {best_trial.params}")
        best_dict = best_trial.user_attrs.copy()
        with open(out_dir / f"optuna_best_config_{suffix}.json", "w") as f:
            json.dump(best_dict, f, indent=4, default=str)
    except ValueError:
        logger.warning("No successful trials completed.")


def run_evaluation_engine(X_panel: pd.DataFrame, y_target: pd.Series, args: argparse.Namespace, out_dir: Path, suffix: str):
    if not args.config_path:
        logger.error("Evaluation mode requires '--config-path'.")
        return

    config_path = Path(args.config_path)
    if not config_path.exists():
        logger.error(f"Config file not found: {args.config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    def _parse_if_str(val):
        if isinstance(val, str):
            try: return json.loads(val)
            except: return val
        return val

    eval_config = {}
    for k, v in config.items():
        if k not in ["rmse", "mae", "status", "runtime_sec", "n_eval_points", "n_folds", "error_message", "n_raw_features", "n_trans_features", "n_sel_features", "n_model_used_vars"]:
            eval_config[k] = _parse_if_str(v)

    logger.info(f"Evaluating DFM with config from {args.config_path}")

    search_grid = [eval_config]
    
    run_signature = build_run_signature(
        script_name="run_dfm_experiment.py",
        input_panel=str(args.input_panel or args.data_dir),
        target=str(args.target),
        seed=args.seed,
        search_last_n_steps=args.search_last_n_steps,
        search_strategy=args.search_strategy,
        engine="evaluate"
    )
    
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pruned_grid = prune_grid(search_grid, args.search_last_n_steps, out_path, resume=args.resume, run_signature=run_signature)
    
    if not pruned_grid:
        logger.info("Evaluation already completed and found in results CSV. Skipping.")
        return

    run_kwargs = {
        "X_panel": X_panel,
        "y_target": y_target,
        "search_mode": args.search_mode,
        "search_last_n_steps": args.search_last_n_steps,
        "scale_x": getattr(args, "scale_x", False)
    }

    results = run_chunks(pruned_grid, run_single_experiment, run_kwargs, out_path, n_jobs=1, chunk_sz=1, stage_name="Evaluation")
    
    if results and results[0]["status"] == "success":
        res = results[0]
        logger.info(f"Evaluation Success | RMSE: {res['rmse']:.4f} | MAE: {res['mae']:.4f}")
        eval_df = res.pop("_eval_df", pd.DataFrame())
        if not eval_df.empty:
            out_preds = out_dir / f"dfm_eval_preds_{suffix}.csv"
            eval_df.to_csv(out_preds, index=False)
            logger.info(f"Saved predictions to {out_preds}")
        
        out_json = out_dir / f"dfm_eval_results_{suffix}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in res.items() if k != "_eval_df"}, f, indent=4, default=str)
        logger.info(f"Saved results to {out_json}")
    else:
        err = results[0].get("error_message") if results else "Unknown error"
        logger.error(f"Evaluation failed: {err}")


def generate_experiment_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    selectors_to_run = [s.strip() for s in args.selectors.split(",")]
    
    k_factors = parse_list(args.dfm_k_factors, int)
    factor_orders = parse_list(args.dfm_factor_orders, int)
    train_windows = parse_list(args.train_windows, int)
    step_sizes = parse_list(args.step_sizes, int)
    window_types = [w.strip() for w in args.window_type.split(",")]
    rolling_window_sizes = parse_list(args.rolling_window_size, int)
    
    grids = []
    
    #Iterate selectors, then build their param variations
    for sel in selectors_to_run:
        param_variations = [{}]
        
        if sel == "pca":
            param_variations = [{"n_components": c} for c in parse_list(args.pca_components, int)]
        elif sel == "factor_analysis":
            param_variations = [{"n_components": c} for c in parse_list(args.factor_analysis_components, int)]
        elif sel == "lasso":
            param_variations = [{"alpha": a} for a in parse_list(args.lasso_alphas, float)]
        elif sel == "elasticnet":
            param_variations = [
                {"alpha": a, "l1_ratio": l} 
                for a in parse_list(args.elasticnet_alphas, float)
                for l in parse_list(args.elasticnet_l1_ratios, float)
            ]
        elif sel == "corr_top_n":
            param_variations = [{"top_n": n} for n in parse_list(args.corr_top_n, int)]
        elif sel == "autoencoder":
            param_variations = [{"latent_dim": n} for n in parse_list(args.ae_latent_dims, int)]
        elif sel == "fast_screen":
            param_variations = [{"top_k": n} for n in parse_list(args.fast_screen_top_k, int)]
            
        for p_var in param_variations:
            for (k, fo, tw, sz, wt) in itertools.product(k_factors, factor_orders, train_windows, step_sizes, window_types):
                cfg = {
                    "selector_method": sel,
                    "selector_params": p_var,
                    "dfm_k_factors": k,
                    "dfm_factor_order": fo,
                    "train_window": tw,
                    "step_size": sz,
                    "window_type": wt,
                    "mixed_frequency": getattr(args, "mixed_frequency", False),
                }
                if getattr(args, "mixed_frequency", False):
                    q_cols = getattr(args, "quarterly_cols", "")
                    cfg["quarterly_cols"] = [c.strip() for c in q_cols.split(",") if c.strip()]
                
                # Expand rolling_window_size only if window_type == "rolling"
                if wt == "rolling":
                    for rws in rolling_window_sizes:
                        r_cfg = cfg.copy()
                        r_cfg["rolling_window_size"] = rws
                        grids.append(r_cfg)
                else:
                    grids.append(cfg)

    if args.search_max_configs > 0 and len(grids) > args.search_max_configs:
        rng = random.Random(getattr(args, "seed", 123))
        rng.shuffle(grids)
        grids = grids[:args.search_max_configs]
        
    return grids


def main():
    ap = build_experiment_parser()
    args = ap.parse_args()
    
    logger.info("=== DFM Experiment Search Runner ===")
    random.seed(args.seed)
    
    # 1. Load Data
    data_dir = Path(args.data_dir)
    if args.mixed_frequency:
        logger.info(f"Loading MF panel ...")
        X_m, X_q, y_target = load_mf_panels(data_dir, args.target, panel_arg=args.input_panel)
        X_panel = pd.concat([X_m, X_q], axis=1) if not X_q.empty else X_m
        if not args.quarterly_cols and not X_q.empty:
            args.quarterly_cols = ",".join(X_q.columns)
        logger.info(f"MF Panel combined shape: {X_panel.shape}, Target shape: {y_target.shape}")
    else:
        logger.info(f"Loading CF panel ...")
        X_panel, y_target = load_cf_panel(data_dir, args.target, panel_arg=args.input_panel)
        logger.info(f"Panel shape: {X_panel.shape}, Target shape: {y_target.shape}")
    
    # 2. Build Grid
    grid = generate_experiment_grid(args)
    logger.info(f"Generated {len(grid)} experiment configurations.")
    
    if len(grid) == 0:
        logger.warning("Empty grid. Exiting.")
        return

    # 3. Parallel Execution
    logger.info(f"Running experiments (n_jobs={args.n_jobs}, mode={args.search_mode}) ...")
    tstart = time.time()
    
    run_kwargs = {
        "X_panel": X_panel,
        "y_target": y_target,
        "search_mode": args.search_mode,
        "search_last_n_steps": args.search_last_n_steps,
        "scale_x": getattr(args, "scale_x", False)
    }
    
    out_path = Path(args.out_file)
    suffix_base = (Path(args.input_panel).stem if args.input_panel else "panel")[:30] + f"_s{args.seed}"
    
    active_engine = args.engine
    if active_engine is None:
        active_engine = "staged" if args.search_strategy == "staged" else "grid"

    if active_engine == "optuna":
        run_optuna_engine(X_panel, y_target, args, out_path.parent, suffix_base)
        return

    if active_engine == "evaluate":
        run_evaluation_engine(X_panel, y_target, args, out_path.parent, suffix_base)
        return

    if active_engine == "staged":
        s1_sels = [s.strip() for s in args.selectors_fast_only.split(",")]
        grid_s1 = [cfg for cfg in grid if cfg["selector_method"] in s1_sels]
        if not grid_s1:
            logger.warning("Stage 1 grid empty (selectors-fast-only mismatch). Falling back to full grid.")
            grid_s1 = grid
            
        run_signature = build_run_signature(
            script_name="run_dfm_experiment.py",
            input_panel=str(args.input_panel or args.data_dir),
            target=str(args.target),
            seed=args.seed,
            search_last_n_steps=args.search_last_n_steps,
            search_strategy=args.search_strategy,
            search_mode=args.search_mode
        )
        
        stage1_steps = min(3, args.search_last_n_steps) if args.search_last_n_steps > 0 else 3
        grid_s1 = prune_grid(grid_s1, stage1_steps, out_path, getattr(args, "resume", False), run_signature=run_signature)
            
        logger.info(f"--- Stage 1: Coarse Search ({len(grid_s1)} configs, {stage1_steps} steps) ---")
        stg1_kwargs = dict(run_kwargs)
        stg1_kwargs["search_last_n_steps"] = stage1_steps
        
        results_s1 = run_chunks(grid_s1, run_single_experiment, stg1_kwargs, out_path, n_jobs=args.n_jobs, chunk_sz=getattr(args, "checkpoint_chunk_size", 10), stage_name="Stage 1")

        if out_path.exists():
            try:
                full_s1_df = pd.read_csv(out_path)
                full_s1_df = full_s1_df[(full_s1_df["status"] == "success")]
                best_configs = full_s1_df.sort_values("rmse").head(args.search_top_k).to_dict("records")
            except Exception:
                df_s1 = pd.DataFrame([r for r in results_s1 if r["status"] == "success"])
                best_configs = df_s1.sort_values("rmse").head(args.search_top_k).to_dict("records") if not df_s1.empty else []
        else:
            df_s1 = pd.DataFrame([r for r in results_s1 if r["status"] == "success"])
            best_configs = df_s1.sort_values("rmse").head(args.search_top_k).to_dict("records") if not df_s1.empty else []

        if not best_configs:
            logger.error("Stage 1 failed completely or produced no results.")
            return

        logger.info(f"--- Stage 2: Fine Search (Top {len(best_configs)} configs, {args.search_last_n_steps} steps) ---")
        clean_keys = set(grid[0].keys())
        staged_grid = [{k: v for k, v in cfg.items() if k in clean_keys} for cfg in best_configs]
        staged_grid = prune_grid(staged_grid, args.search_last_n_steps, out_path, getattr(args, "resume", False), run_signature=run_signature)
        
        results = run_chunks(staged_grid, run_single_experiment, run_kwargs, out_path, n_jobs=args.n_jobs, chunk_sz=getattr(args, "checkpoint_chunk_size", 10), stage_name="Stage 2")
    else:
        run_signature = build_run_signature(
            script_name="run_dfm_experiment.py",
            input_panel=str(args.input_panel or args.data_dir),
            target=str(args.target),
            seed=args.seed,
            search_last_n_steps=args.search_last_n_steps,
            search_strategy=args.search_strategy,
            search_mode=args.search_mode
        )
        grid = prune_grid(grid, args.search_last_n_steps, out_path, getattr(args, "resume", False), run_signature=run_signature)
        results = run_chunks(grid, run_single_experiment, run_kwargs, out_path, n_jobs=args.n_jobs, chunk_sz=getattr(args, "checkpoint_chunk_size", 10))
            
    # 4. Collect and save results
    total_time = time.time() - tstart
    
    if out_path.exists():
        df_res = pd.read_csv(out_path)
    else:
        df_res = pd.DataFrame(results)
    
    success = df_res[df_res["status"] == "success"]
    failed_count = len(df_res) - len(success)
    
    if not success.empty:
        df_res = df_res.sort_values("rmse", ascending=True).reset_index(drop=True)
        
        logger.info(f"\nExperiment complete in {total_time/60:.1f} minutes.")
        logger.info(f"Success: {len(success)}, Failed: {failed_count}")
        logger.info("\n--- Top 5 Configurations by RMSE ---")
        
        print(df_res.head(5)[["selector_method", "selector_params", "dfm_k_factors", "train_window", "rmse", "mae", "runtime_sec"]].to_string())
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_res.to_csv(out_path, index=False)
        logger.info(f"\nDetailed results saved to {out_path.absolute()}")

        # Best params re-eval to generate prediction output if _eval_df was lost from memory
        best_cfg = df_res.iloc[0].to_dict()
        memory_res = [r for r in results if r.get("_config_id") == best_cfg.get("_config_id")]
        if memory_res and "_eval_df" in memory_res[0]:
            best_eval_df = memory_res[0]["_eval_df"]
        else:
            logger.info("Re-evaluating best configuration to regenerate predictions output...")
            b_cfg = {k:v for k,v in best_cfg.items() if not str(k).startswith("_")}
            if isinstance(b_cfg.get('selector_params'), str):
                b_cfg['selector_params'] = json.loads(b_cfg['selector_params'])
            b_res = run_single_experiment(b_cfg, X_panel, y_target, args.search_mode, args.search_last_n_steps, scale_x=getattr(args, "scale_x", False))
            best_eval_df = b_res.get("_eval_df", pd.DataFrame())

        if not best_eval_df.empty:
            preds_path = out_path.parent / f"predictions_dfm_{args.seed}.csv"
            best_eval_df.to_csv(preds_path, index=False)
            logger.info(f"DFM best preds → {preds_path}")

    else:
        logger.error("ALL experiments failed. Check logs.")

if __name__ == "__main__":
    main()
