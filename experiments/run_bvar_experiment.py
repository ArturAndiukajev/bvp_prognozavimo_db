"""
run_bvar_experiment.py - BVAR & VAR Grid Search Experiment Pipeline
====================================================================

Runs a systematic grid search over VAR/BVAR hyperparameters and feature
selection strategies. Supports optional parallel execution via joblib.

Example (quick smoke test):
    python scripts/run_bvar_experiment.py \\
        --modes bvar,var \\
        --lags 1,2,3 \\
        --lambda1 0.01,0.1,1.0 \\
        --lambda2 0.5,1.0 \\
        --lambda3 1,2 \\
        --selectors none,pca \\
        --pca-components 5,10 \\
        --train-windows 80,120 \\
        --n-jobs 4 \\
        --seed 123

Full run replacing all defaults:
    python scripts/run_bvar_experiment.py \\
        --modes bvar,var \\
        --lags 1,2,3,4,6 \\
        --lambda1 0.01,0.1,1.0 \\
        --lambda2 0.1,0.5,1.0 \\
        --lambda3 1,2 \\
        --selectors none,pca,corr_top_n,lasso,elasticnet \\
        --pca-components 3,5,10 \\
        --corr-top-n 10,20,50 \\
        --lasso-alphas 0.01,0.1 \\
        --elasticnet-alphas 0.01,0.1 \\
        --elasticnet-l1-ratios 0.2,0.5,0.8 \\
        --train-windows 60,80,120 \\
        --step-sizes 1,3 \\
        --n-jobs 4 \\
        --seed 123
"""

import argparse
import itertools
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Force single-thread BLAS in parallel workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Ensure nowcasting package is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nowcasting.main import load_cf_panel, _DEFAULT_DATA_DIR
from nowcasting.evaluation.backtester import RollingBacktester
from nowcasting.evaluation.metrics import compute_metrics
from nowcasting.models.bvar import BVARNowcast

from nowcasting.features.selectors import (
    IdentitySelector, VarianceFilter, PCACompressor, FactorAnalysisCompressor,
    LassoSelector, ElasticNetSelector, CorrTopNSelector, FastScreeningFilter
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("bvar_experiment")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_experiment_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="BVAR / VAR Feature Selection Experiment Runner")

    # Data
    ap.add_argument("--data-dir", type=str, default=_DEFAULT_DATA_DIR)
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--input-panel", type=str, default=None)

    # Mode
    ap.add_argument("--modes", type=str, default="bvar,var",
                    help="Comma-separated: bvar, var")

    # VAR / BVAR hyperparams
    ap.add_argument("--lags",    type=str, default="1,2,3,4",
                    help="Comma-separated lag orders")
    ap.add_argument("--lambda1", type=str, default="0.01,0.1,1.0",
                    help="Overall shrinkage (BVAR only)")
    ap.add_argument("--lambda2", type=str, default="0.1,0.5,1.0",
                    help="Cross-variable shrinkage (BVAR only)")
    ap.add_argument("--lambda3", type=str, default="1,2",
                    help="Lag-decay exponents (BVAR only)")
    ap.add_argument("--max-vars", type=int, default=30,
                    help="Maximum variables in the VAR system (dimensionality cap)")

    # Feature selectors
    ap.add_argument("--selectors", type=str, default="none,fast_screen,pca,corr_top_n",
                    help="Comma-separated: none, fast_screen, pca, corr_top_n, lasso, elasticnet, factor_analysis")
    ap.add_argument("--pca-components",        type=str, default="3,5,10")
    ap.add_argument("--factor-analysis-components", type=str, default="3,5")
    ap.add_argument("--corr-top-n",            type=str, default="10,20,50")
    ap.add_argument("--lasso-alphas",          type=str, default="0.01,0.1")
    ap.add_argument("--elasticnet-alphas",     type=str, default="0.01,0.1")
    ap.add_argument("--elasticnet-l1-ratios",  type=str, default="0.5")
    ap.add_argument("--fast-screen-top-k",     type=str, default="50,100")


    # Training settings
    ap.add_argument("--train-windows", type=str, default="60,80,120",
                    help="Initial training window sizes (comma-separated)")
    ap.add_argument("--step-sizes",    type=str, default="1",
                    help="Backtest step sizes (comma-separated)")

    # Search controls
    ap.add_argument("--search-last-n-steps", type=int, default=0,
                    help="Only run the last N backtest steps (0 = all).")
    ap.add_argument("--search-max-configs",  type=int, default=0,
                    help="Randomly sub-sample at most this many configs (0 = no limit).")
    ap.add_argument("--search-strategy", type=str, default="full", choices=["full", "staged"],
                    help="Use 'staged' for a fast 1-pass coarse search to filter options.")
    ap.add_argument("--search-top-k", type=int, default=5,
                    help="Top configs to promote to final pass in staged search.")

    # Compute
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--seed",   type=int, default=123,
                    help="Global random seed for reproducibility.")

    # Output
    ap.add_argument("--out-file", type=str,
                    default="data/forecasts/bvar_experiment_results.csv")

    return ap


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_list(val: str, dtype=str) -> list:
    return [dtype(v.strip()) for v in val.split(",") if v.strip()]


def build_selector(method: str, params: dict):
    """Construct a feature selector / compressor from method name + params."""
    if method == "none":
        return IdentitySelector()
    elif method == "pca":
        return PCACompressor(n_components=params.get("n_components", 5))
    elif method == "factor_analysis":
        return FactorAnalysisCompressor(n_components=params.get("n_components", 5))
    elif method == "lasso":
        return LassoSelector(alpha=params.get("alpha", 0.1))
    elif method == "elasticnet":
        return ElasticNetSelector(alpha=params.get("alpha", 0.1),
                                  l1_ratio=params.get("l1_ratio", 0.5))
    elif method == "corr_top_n":
        return CorrTopNSelector(top_n=params.get("top_n", 20))
    elif method == "fast_screen":
        return FastScreeningFilter(top_k=params.get("top_n", 50))
    else:
        raise ValueError(f"Unknown selector method: '{method}'")


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def generate_experiment_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    modes            = parse_list(args.modes, str)
    lags_list        = parse_list(args.lags, int)
    lambda1_list     = parse_list(args.lambda1, float)
    lambda2_list     = parse_list(args.lambda2, float)
    lambda3_list     = parse_list(args.lambda3, float)
    train_windows    = parse_list(args.train_windows, int)
    step_sizes       = parse_list(args.step_sizes, int)
    selectors_to_run = parse_list(args.selectors, str)

    grids: List[Dict[str, Any]] = []

    for mode in modes:
        # Minnesota parameters only apply to BVAR
        if mode == "bvar":
            prior_combos = list(itertools.product(lambda1_list, lambda2_list, lambda3_list))
        else:
            prior_combos = [(None, None, None)]   # one placeholder row for VAR

        for sel in selectors_to_run:
            param_variations = [{}]
            if sel == "pca":
                param_variations = [{"n_components": c}
                                    for c in parse_list(args.pca_components, int)]
            elif sel == "factor_analysis":
                param_variations = [{"n_components": c}
                                    for c in parse_list(args.factor_analysis_components, int)]
            elif sel == "corr_top_n":
                param_variations = [{"top_n": n}
                                    for n in parse_list(args.corr_top_n, int)]
            elif sel == "lasso":
                param_variations = [{"alpha": a}
                                    for a in parse_list(args.lasso_alphas, float)]
            elif sel == "elasticnet":
                param_variations = [
                    {"alpha": a, "l1_ratio": l}
                    for a in parse_list(args.elasticnet_alphas, float)
                    for l in parse_list(args.elasticnet_l1_ratios, float)
                ]
            elif sel == "fast_screen":
                param_variations = [{"top_n": n}
                                    for n in parse_list(args.fast_screen_top_k, int)]

            for p_var in param_variations:
                for (l1, l2, l3) in prior_combos:
                    for (lags, tw, sz) in itertools.product(lags_list, train_windows, step_sizes):
                        grids.append({
                            "mode":             mode,
                            "lags":             lags,
                            "lambda1":          l1,
                            "lambda2":          l2,
                            "lambda3":          l3,
                            "selector_method":  sel,
                            "selector_params":  p_var,
                            "train_window":     tw,
                            "step_size":        sz,
                        })

    # Optional random sub-sampling
    if args.search_max_configs > 0 and len(grids) > args.search_max_configs:
        random.shuffle(grids)
        grids = grids[:args.search_max_configs]

    return grids


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    config: Dict[str, Any],
    X_panel: pd.DataFrame,
    y_target: pd.Series,
    search_last_n_steps: int,
    max_vars: int,
    seed: int,
) -> dict:
    """Execute one grid configuration and return a metrics dict."""
    res = {**config, "status": "running"}
    # Serialise selector_params for JSON safety
    res["selector_params"] = json.dumps(config["selector_params"])
    res["window_type"] = "expanding"
    res["eval_mode"] = "common_frequency"
    res["n_folds"] = None

    np.random.seed(seed)   # worker-level determinism

    try:
        t0 = time.time()

        # --- Backtester ---
        init_train = config["train_window"]
        step_sz    = config["step_size"]
        if search_last_n_steps > 0:
            required_start = len(y_target) - search_last_n_steps
            init_train = max(init_train, required_start)

        bt = RollingBacktester(
            initial_train_periods=init_train, 
            step_size=step_sz,
            window_type="expanding",
            eval_mode="common_frequency"
        )

        # --- Selector pipeline ---
        from sklearn.pipeline import make_pipeline
        base_sel = build_selector(config["selector_method"], config["selector_params"])
        selector = make_pipeline(VarianceFilter(), base_sel)

        # --- Model ---
        mode = config["mode"]
        lags = config["lags"]

        if mode == "bvar":
            model = BVARNowcast(
                target_col=str(y_target.name),
                mode="bvar",
                lags=lags,
                lambda1=config["lambda1"],
                lambda2=config["lambda2"],
                lambda3=config["lambda3"],
                max_vars=max_vars,
                seed=seed,
            )
        else:
            model = BVARNowcast(
                target_col=str(y_target.name),
                mode="var",
                lags=lags,
                max_vars=max_vars,
                seed=seed,
            )

        # --- Backtest ---
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eval_df = bt.backtest(
                model=model,
                X_panel=X_panel,
                y_target=y_target,
                transformer=None,
                feature_selector=selector,
            )

        res["runtime_sec"] = round(time.time() - t0, 2)

        if eval_df.empty:
            res["status"] = "failed"
            res["error_message"] = "Backtester returned empty DataFrame"
            return res

        metrics = compute_metrics(eval_df, model_name=f"BVAR_{mode}")
        res["rmse"]          = float(metrics["RMSE"].iloc[0])
        res["mae"]           = float(metrics["MAE"].iloc[0])
        res["mape"]          = float(metrics["MAPE"].iloc[0]) if "MAPE" in metrics.columns else float("nan")
        res["n_eval_points"] = len(eval_df)
        res["n_folds"]       = eval_df["Forecast_Origin"].nunique() if "Forecast_Origin" in eval_df.columns else len(eval_df)
        
        # Track feature counts seamlessly natively provided by the backtester updates
        res["n_raw_features"]    = int(eval_df["n_raw_features"].median()) if "n_raw_features" in eval_df else None
        res["n_trans_features"]  = int(eval_df["n_trans_features"].median()) if "n_trans_features" in eval_df else None
        res["n_sel_features"]    = int(eval_df["n_sel_features"].median()) if "n_sel_features" in eval_df else None
        res["n_model_used_vars"] = int(eval_df["n_model_used_features"].median()) if "n_model_used_features" in eval_df else None
        
        res["status"]        = "success"
        res["_eval_df"]      = eval_df

    except Exception as exc:
        logger.warning(f"Experiment failed: {config} → {exc}")
        res["status"]        = "failed"
        res["error_message"] = str(exc)
        res.setdefault("runtime_sec", round(time.time() - t0, 2))

    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap   = build_experiment_parser()
    args = ap.parse_args()

    # Seed everything at the top level
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=== BVAR / VAR Experiment Search Runner ===")
    logger.info(f"Seed: {args.seed}  |  n_jobs: {args.n_jobs}")

    # 1. Load data
    data_dir = Path(args.data_dir)
    logger.info("Loading panel ...")
    X_panel, y_target = load_cf_panel(data_dir, args.target, panel_arg=args.input_panel)
    logger.info(f"Panel shape: {X_panel.shape}  |  Target: {y_target.name}")

    # 2. Build grid
    grid = generate_experiment_grid(args)
    logger.info(f"Generated {len(grid)} experiment configurations.")
    if not grid:
        logger.warning("Empty grid — exiting.")
        return

    # 3. Run
    logger.info(f"Running experiments ...")
    tstart = time.time()

    run_kwargs = dict(
        X_panel=X_panel,
        y_target=y_target,
        search_last_n_steps=args.search_last_n_steps,
        max_vars=args.max_vars,
        seed=args.seed,
    )

    if args.search_strategy == "staged":
        logger.info(f"--- Stage 1: Coarse Search ({len(grid)} configs, 3 steps) ---")
        run_kwargs["search_last_n_steps"] = min(3, args.search_last_n_steps) if args.search_last_n_steps > 0 else 3
        if args.n_jobs > 1:
            results_s1 = Parallel(n_jobs=args.n_jobs, verbose=5)(
                delayed(run_single_experiment)(cfg, **run_kwargs) for cfg in grid
            )
        else:
            results_s1 = [run_single_experiment(cfg, **run_kwargs) for cfg in grid]

        df_s1 = pd.DataFrame([r for r in results_s1 if r["status"] == "success"])
        if df_s1.empty:
            logger.error("Stage 1 failed completely.")
            return

        df_s1 = df_s1.sort_values("rmse")
        top_k = min(len(df_s1), args.search_top_k)
        best_configs = df_s1.head(top_k).to_dict("records")

        logger.info(f"--- Stage 2: Fine Search (Top {top_k} configs, {args.search_last_n_steps} steps) ---")
        clean_keys = set(grid[0].keys())
        staged_grid = [{k: v for k, v in cfg.items() if k in clean_keys} for cfg in best_configs]
        
        run_kwargs["search_last_n_steps"] = args.search_last_n_steps
        if args.n_jobs > 1:
            results = Parallel(n_jobs=args.n_jobs, verbose=5)(
                delayed(run_single_experiment)(cfg, **run_kwargs) for cfg in staged_grid
            )
        else:
            results = [run_single_experiment(cfg, **run_kwargs) for cfg in staged_grid]
    else:
        if args.n_jobs > 1:
            results = Parallel(n_jobs=args.n_jobs, verbose=5)(
                delayed(run_single_experiment)(cfg, **run_kwargs) for cfg in grid
            )
        else:
            results = []
            for i, cfg in enumerate(grid, 1):
                logger.info(f"[{i}/{len(grid)}] mode={cfg['mode']:4s}  lags={cfg['lags']}"
                            f"  l1={cfg['lambda1']}  sel={cfg['selector_method']}"
                            f"  tw={cfg['train_window']}")
                results.append(run_single_experiment(cfg, **run_kwargs))

    total_time = time.time() - tstart

    # 4. Collect results
    df_res = pd.DataFrame(results)
    df_res["dataset"] = str(args.input_panel or args.data_dir)
    df_res["seed"]    = args.seed

    success      = df_res[df_res["status"] == "success"]
    failed_count = len(df_res) - len(success)

    logger.info(f"\nExperiment complete in {total_time / 60:.1f} min  |  "
                f"Success: {len(success)}  |  Failed: {failed_count}")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not success.empty:
        df_res = df_res.sort_values("rmse", ascending=True).reset_index(drop=True)
        logger.info("\n--- Top 5 Configurations by RMSE ---")
        display_cols = [c for c in ["mode", "lags", "lambda1", "lambda2", "selector_method",
                                     "train_window", "rmse", "mae", "runtime_sec"]
                        if c in df_res.columns]
        print(df_res.head(5)[display_cols].to_string(index=False))

        best_idx = success["rmse"].idxmin()
        best_eval_df = df_res.loc[best_idx, "_eval_df"]
        preds_path = out_path.parent / f"predictions_bvar_{args.seed}.csv"
        best_eval_df.to_csv(preds_path, index=False)
        logger.info(f"BVAR best preds → {preds_path}")

        if "_eval_df" in df_res.columns:
            df_res = df_res.drop(columns=["_eval_df"])

        df_res.to_csv(out_path, index=False)
        logger.info(f"\nDetailed results saved to: {out_path.absolute()}")

        # Best configuration JSON
        best = df_res.iloc[0].to_dict()
        if "_eval_df" in best:
            del best["_eval_df"]
        try:
            best["selector_params"] = json.loads(best["selector_params"])
        except Exception:
            pass
        best_path = out_path.parent / "best_bvar_configuration.json"
        with open(best_path, "w") as fh:
            json.dump(best, fh, indent=4, default=str)
        logger.info(f"Best configuration saved to: {best_path}")
    else:
        logger.error("ALL experiments failed. Saving failure log.")
        if "_eval_df" in df_res.columns:
            df_res = df_res.drop(columns=["_eval_df"])
        df_res.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
