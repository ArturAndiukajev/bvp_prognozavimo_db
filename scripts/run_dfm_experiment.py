"""
run_dfm_experiment.py - DFM Grid Search & Feature Selection Experiment Pipeline
=============================================================================

Executes a grid search over DFM hyperparameters and feature selection
strategies. Parallelizes outer loops to speed up backtesting. Resulting
combinations are sorted by RMSE/MAE and exported to CSV.

Example:
    python scripts/run_dfm_experiment.py --selectors none,pca,lasso,elasticnet \
           --dfm-k-factors 1,2,3 \
           --pca-components 3,5 \
           --lasso-alphas 0.05,0.1 \
           --n-jobs 4 --search-mode quick
"""

import argparse
import itertools
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from joblib import Parallel, delayed

import pandas as pd
import numpy as np

# Force single-thread linear algebra to prevent collision in parallel workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# We import pipeline bits from nowcasting package
from nowcasting.main import build_arg_parser as build_main_parser, load_cf_panel, _DEFAULT_DATA_DIR
from nowcasting.evaluation.backtester import RollingBacktester
from nowcasting.evaluation.metrics import compute_metrics
from nowcasting.models.dfm import DynamicFactorNowcast

from nowcasting.features.selectors import (
    IdentitySelector, VarianceFilter, PCACompressor, FactorAnalysisCompressor,
    LassoSelector, ElasticNetSelector, CorrTopNSelector, AutoencoderCompressor
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("dfm_experiment")


def build_experiment_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="DFM Feature Selection Experiment Runner")
    
    # Data arguments 
    ap.add_argument("--data-dir", type=str, default=_DEFAULT_DATA_DIR)
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--input-panel", type=str, default=None)
    
    # Grid Search Spaces (Comma-separated lists)
    ap.add_argument("--selectors", type=str, default="none,pca,lasso,elasticnet,corr_top_n",
                    help="Comma-separated selectors to test: none,pca,lasso,elasticnet,corr_top_n,factor_analysis,autoencoder")
    
    ap.add_argument("--dfm-k-factors", type=str, default="1,2,3")
    ap.add_argument("--dfm-factor-orders", type=str, default="1,2")
    
    ap.add_argument("--pca-components", type=str, default="3,5,10")
    ap.add_argument("--factor-analysis-components", type=str, default="3,5")
    ap.add_argument("--lasso-alphas", type=str, default="0.01,0.1")
    ap.add_argument("--elasticnet-alphas", type=str, default="0.01,0.1")
    ap.add_argument("--elasticnet-l1-ratios", type=str, default="0.5")
    ap.add_argument("--corr-top-n", type=str, default="10,20,50")
    ap.add_argument("--ae-latent-dims", type=str, default="3,5")

    # Time rules
    ap.add_argument("--train-windows", type=str, default="120", 
                    help="Initial train window sizes (e.g., 80,120)")
    ap.add_argument("--step-sizes", type=str, default="1", 
                    help="Backtest step sizes (e.g., 1,3)")
    
    # Experiment controls
    ap.add_argument("--search-mode", type=str, choices=["quick", "full"], default="full",
                    help="Quick mode reduces maxiter and uses fewer backtest steps.")
    ap.add_argument("--search-last-n-steps", type=int, default=0,
                    help="Only run the backtest for the last N steps to save time.")
    ap.add_argument("--search-max-configs", type=int, default=0,
                    help="Limit the number of random configurations tested.")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Number of parallel workers for joblib.")
    
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
    else:
        raise ValueError(f"Unknown selector method: {method}")


def run_single_experiment(
    config: Dict[str, Any],
    X_panel: pd.DataFrame, 
    y_target: pd.Series,
    search_mode: str,
    search_last_n_steps: int
) -> dict:
    """Runs a single combination grid configuration and returns metrics."""
    res = {**config}
    res["status"] = "running"
    res["window_type"] = "expanding"
    res["eval_mode"] = "common_frequency"
    res["n_folds"] = None
    
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
            window_type="expanding",
            eval_mode="common_frequency"
        )

        
        # 2. Selector Initialization
        # VarianceFilter is always implicitly added in front to drop constants safely
        from sklearn.pipeline import make_pipeline
        base_sel = build_selector(config["selector_method"], config["selector_params"])
        selector = make_pipeline(VarianceFilter(), base_sel)
        
        # 3. Model Initialization
        k = config["dfm_k_factors"]
        fo = config["dfm_factor_order"]
        mod_kws = {}
        if search_mode == "quick":
            mod_kws["maxiter"] = 50  # Faster, lower tolerance DFM fits
            
        model = DynamicFactorNowcast(target_col=str(y_target.name), k_factors=k, factor_order=fo, **mod_kws)
        
        # 4. Run loop
        # We temporarily disable internal statsmodels progress output inside parallel steps
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eval_df = bt.backtest(
                model=model,
                X_panel=X_panel,
                y_target=y_target,
                transformer=None,  # Not used, but could add feature engineering here
                feature_selector=selector
            )
            
        runt = time.time() - t0
        res["runtime_sec"] = round(runt, 2)
        
        if eval_df.empty:
            res["status"] = "failed"
            res["error_message"] = "Backtester returned empty DataFrame"
            return res
            
        # Extract metadata counts from final pipeline state if available
        # Actually, extracting feature counts from backtester is hard since pipeline is fit internally per step. 
        # But we can approximate using final model features
        metrics = compute_metrics(eval_df, model_name=f"DFM_{k}f")
        res["rmse"] = metrics["RMSE"].iloc[0]
        res["mae"] = metrics["MAE"].iloc[0]
        res["status"] = "success"
        res["n_eval_points"] = len(eval_df)
        res["n_folds"]       = eval_df["Forecast_Origin"].nunique() if "Forecast_Origin" in eval_df.columns else len(eval_df)

        
    except Exception as e:
        logger.warning(f"Experiment failed: {config} -> {e}")
        res["status"] = "failed"
        res["error_message"] = str(e)
        
    return res


def generate_experiment_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    selectors_to_run = [s.strip() for s in args.selectors.split(",")]
    
    k_factors = parse_list(args.dfm_k_factors, int)
    factor_orders = parse_list(args.dfm_factor_orders, int)
    train_windows = parse_list(args.train_windows, int)
    step_sizes = parse_list(args.step_sizes, int)
    
    grids = []
    
    # We iterate selectors, then build their param variations
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
            
        for p_var in param_variations:
            # Cartesian with DFM params
            for (k, fo, tw, sz) in itertools.product(k_factors, factor_orders, train_windows, step_sizes):
                grids.append({
                    "selector_method": sel,
                    "selector_params": p_var,
                    "dfm_k_factors": k,
                    "dfm_factor_order": fo,
                    "train_window": tw,
                    "step_size": sz
                })
                
    if args.search_max_configs > 0 and len(grids) > args.search_max_configs:
        import random
        random.shuffle(grids)
        grids = grids[:args.search_max_configs]
        
    return grids


def main():
    ap = build_experiment_parser()
    args = ap.parse_args()
    
    logger.info("=== DFM Experiment Search Runner ===")
    
    # 1. Load Data
    data_dir = Path(args.data_dir)
    logger.info(f"Loading panel ...")
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
        "search_last_n_steps": args.search_last_n_steps
    }
    
    if args.n_jobs > 1:
        # Joblib inner executor
        results = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(run_single_experiment)(cfg, **run_kwargs) for cfg in grid
        )
    else:
        results = []
        for i, cfg in enumerate(grid, 1):
            logger.info(f"[{i}/{len(grid)}] Running: {cfg}")
            results.append(run_single_experiment(cfg, **run_kwargs))
            
    # 4. Collect and save results
    total_time = time.time() - tstart
    df_res = pd.DataFrame(results)
    
    # Format the dict columns for CSV readability
    df_res['selector_params'] = df_res['selector_params'].apply(lambda x: json.dumps(x))
    
    success = df_res[df_res["status"] == "success"]
    failed_count = len(df_res) - len(success)
    
    if not success.empty:
        df_res = df_res.sort_values("rmse", ascending=True).reset_index(drop=True)
        
        logger.info(f"\nExperiment complete in {total_time/60:.1f} minutes.")
        logger.info(f"Success: {len(success)}, Failed: {failed_count}")
        logger.info("\n--- Top 5 Configurations by RMSE ---")
        
        print(df_res.head(5)[["selector_method", "selector_params", "dfm_k_factors", "train_window", "rmse", "mae", "runtime_sec"]].to_string())
        
        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_res.to_csv(out_path, index=False)
        logger.info(f"\nDetailed results saved to {out_path.absolute()}")
    else:
        logger.error("ALL experiments failed. Check logs.")
        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_res.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
