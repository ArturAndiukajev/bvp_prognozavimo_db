"""
run_bridge_search.py - Bridge Equation Grid Search & Feature Selection Experiment Pipeline
==========================================================================================

Executes a grid search over Bridge Equation hyperparameters, regression models,
and feature selection strategies. Parallelizes outer loops to speed up backtesting.
Resulting combinations are sorted by RMSE/MAE and exported to CSV.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List

from joblib import Parallel, delayed
import pandas as pd

# Force single-thread linear algebra to prevent collision in parallel workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from nowcasting.main import load_cf_panel, load_mf_panels, _DEFAULT_DATA_DIR
from nowcasting.evaluation.backtester import RollingBacktester
from nowcasting.evaluation.metrics import compute_metrics
from nowcasting.models.bridge_equation import BridgeEquationNowcast
from nowcasting.features.selectors import (
    IdentitySelector,
    VarianceFilter,
    PCACompressor,
    FactorAnalysisCompressor,
    LassoSelector,
    ElasticNetSelector,
    CorrTopNSelector,
    AutoencoderCompressor,
    FastScreeningFilter,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("bridge_experiment")


def build_experiment_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Bridge Feature Selection Experiment Runner")

    # Data arguments
    ap.add_argument("--data-dir", type=str, default=_DEFAULT_DATA_DIR)
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--input-panel", type=str, default=None)
    ap.add_argument(
        "--mixed-frequency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run in mixed-frequency mode (default: True for Bridge).",
    )
    ap.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")

    # Grid Search Spaces
    ap.add_argument(
        "--selectors",
        type=str,
        default="none,fast_screen,pca,lasso,elasticnet,corr_top_n",
        help="Comma-separated selectors to test.",
    )

    # Bridge-specific
    ap.add_argument("--ar-lags", type=str, default="1,2,3")
    ap.add_argument("--agg-rules", type=str, default="mean,last,sum")
    ap.add_argument("--regression-models", type=str, default="linear,ridge,lasso,elasticnet")

    # Feature selector hyperparams
    ap.add_argument("--pca-components", type=str, default="3,5,10")
    ap.add_argument("--factor-analysis-components", type=str, default="3,5")
    ap.add_argument("--lasso-alphas", type=str, default="0.01,0.1")
    ap.add_argument("--elasticnet-alphas", type=str, default="0.01,0.1")
    ap.add_argument("--elasticnet-l1-ratios", type=str, default="0.5")
    ap.add_argument("--corr-top-n", type=str, default="10,20,50")
    ap.add_argument("--ae-latent-dims", type=str, default="3,5")
    ap.add_argument("--fast-screen-top-k", type=str, default="50,100")

    # Time rules
    ap.add_argument("--train-windows", type=str, default="120", help="Initial train window sizes.")
    ap.add_argument("--step-sizes", type=str, default="1", help="Backtest step sizes.")
    ap.add_argument(
        "--window-type",
        type=str,
        default="expanding",
        help="Comma-separated: expanding, rolling, or both.",
    )
    ap.add_argument(
        "--rolling-window-size",
        type=str,
        default="120",
        help="Comma-separated rolling window sizes.",
    )

    # Experiment controls
    ap.add_argument("--search-mode", type=str, choices=["quick", "full"], default="full")
    ap.add_argument("--search-last-n-steps", type=int, default=0)
    ap.add_argument("--search-max-configs", type=int, default=0)
    ap.add_argument("--search-strategy", type=str, default="full", choices=["full", "staged"])
    ap.add_argument("--search-top-k", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=1)

    ap.add_argument("--out-file", type=str, default="data/forecasts/bridge_search_results.csv")
    return ap


def parse_list(val_str: str, dtype=str):
    return [dtype(v.strip()) for v in val_str.split(",") if v.strip()]


def build_selector(method: str, params: dict):
    if method == "none":
        return IdentitySelector()
    if method == "pca":
        return PCACompressor(n_components=params.get("n_components", 5))
    if method == "factor_analysis":
        return FactorAnalysisCompressor(n_components=params.get("n_components", 5))
    if method == "lasso":
        return LassoSelector(alpha=params.get("alpha", 0.1))
    if method == "elasticnet":
        return ElasticNetSelector(
            alpha=params.get("alpha", 0.1),
            l1_ratio=params.get("l1_ratio", 0.5),
        )
    if method == "corr_top_n":
        return CorrTopNSelector(top_n=params.get("top_n", 20))
    if method == "autoencoder":
        return AutoencoderCompressor(latent_dim=params.get("latent_dim", 5), epochs=10)
    if method == "fast_screen":
        return FastScreeningFilter(top_k=params.get("top_k", 50))
    raise ValueError(f"Unknown selector method: {method}")


def _load_bridge_data(args: argparse.Namespace):
    data_dir = Path(args.data_dir)

    if not args.mixed_frequency:
        logger.info("Loading CF panel ...")
        return load_cf_panel(data_dir, args.target, panel_arg=args.input_panel)

    logger.info("Loading MF panel ...")
    loaded = load_mf_panels(data_dir, args.target, panel_arg=args.input_panel)

    # Flexible unpacking for project-specific loader variants
    if isinstance(loaded, tuple):
        if len(loaded) == 3:
            X_monthly, X_quarterly, y_target = loaded
        elif len(loaded) == 2:
            X_monthly, y_target = loaded
            X_quarterly = pd.DataFrame()
        else:
            raise ValueError("load_mf_panels returned an unsupported tuple shape.")
    elif isinstance(loaded, dict):
        X_monthly = loaded.get("X_monthly", loaded.get("X_m", pd.DataFrame()))
        X_quarterly = loaded.get("X_quarterly", loaded.get("X_q", pd.DataFrame()))
        y_target = loaded.get("y_target", loaded.get("target"))
        if y_target is None:
            raise ValueError("load_mf_panels dict must include y_target.")
    else:
        raise ValueError("load_mf_panels returned an unsupported object.")

    # BridgeEquationNowcast expects the high-frequency panel as X.
    if X_monthly is not None and not X_monthly.empty:
        X_panel = X_monthly
    else:
        raise ValueError("BridgeEquationNowcast requires a non-empty high-frequency panel.")

    return X_panel, y_target


def run_single_experiment(
    config: Dict[str, Any],
    X_panel: pd.DataFrame,
    y_target: pd.Series,
    search_mode: str,
    search_last_n_steps: int,
) -> dict:
    res = {**config}
    res["status"] = "running"
    res["window_type"] = config.get("window_type", "expanding")
    res["eval_mode"] = "mixed_frequency" if config.get("mixed_frequency", True) else "common_frequency"
    res["n_folds"] = None

    try:
        t0 = time.time()

        init_train = config["train_window"]
        step_sz = config["step_size"]

        if search_last_n_steps > 0:
            # In MF mode, y_target length is the relevant count of LF target observations.
            required_start_index = max(0, len(y_target.dropna()) - search_last_n_steps)
            init_train = max(init_train, required_start_index)

        bt = RollingBacktester(
            initial_train_periods=init_train,
            step_size=step_sz,
            window_type=config.get("window_type", "expanding"),
            eval_mode=res["eval_mode"],
            rolling_window_size=config.get("rolling_window_size"),
        )

        from sklearn.pipeline import make_pipeline

        base_sel = build_selector(config["selector_method"], config["selector_params"])
        selector = make_pipeline(VarianceFilter(), base_sel)

        mod_kws = {}
        if search_mode == "quick" and config["regression_model"] in {"lasso", "elasticnet"}:
            mod_kws["max_iter"] = 1000

        model = BridgeEquationNowcast(
            target_col=str(y_target.name),
            ar_lags=config["ar_lags"],
            agg_rule=config["agg_rule"],
            regression_model=config["regression_model"],
            regression_kwargs=mod_kws,
            random_state=config.get("seed", 123),
        )

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

        metrics = compute_metrics(eval_df, model_name="Bridge")

        res["rmse"] = metrics["RMSE"].iloc[0]
        res["mae"] = metrics["MAE"].iloc[0]
        res["status"] = "success"
        res["_eval_df"] = eval_df
        res["n_eval_points"] = len(eval_df)
        res["n_folds"] = (
            eval_df["Forecast_Origin"].nunique()
            if "Forecast_Origin" in eval_df.columns
            else len(eval_df)
        )

        res["n_raw_features"] = (
            int(eval_df["n_raw_features"].median()) if "n_raw_features" in eval_df.columns else None
        )
        res["n_trans_features"] = (
            int(eval_df["n_trans_features"].median()) if "n_trans_features" in eval_df.columns else None
        )
        res["n_sel_features"] = (
            int(eval_df["n_sel_features"].median()) if "n_sel_features" in eval_df.columns else None
        )
        res["n_model_used_vars"] = (
            int(eval_df["n_model_used_features"].median())
            if "n_model_used_features" in eval_df.columns
            else None
        )
    except Exception as e:
        logger.warning(f"Experiment failed: {config} -> {e}")
        res["status"] = "failed"
        res["error_message"] = str(e)

    return res


def generate_experiment_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    selectors_to_run = parse_list(args.selectors, str)
    ar_lags_list = parse_list(args.ar_lags, int)
    agg_rules = parse_list(args.agg_rules, str)
    regression_models = parse_list(args.regression_models, str)
    train_windows = parse_list(args.train_windows, int)
    step_sizes = parse_list(args.step_sizes, int)
    window_types = [w.strip() for w in args.window_type.split(",") if w.strip()]
    rolling_window_sizes = parse_list(args.rolling_window_size, int)

    grids: List[Dict[str, Any]] = []

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
            for lags, agg, reg_m, tw, sz, wt in itertools.product(
                ar_lags_list,
                agg_rules,
                regression_models,
                train_windows,
                step_sizes,
                window_types,
            ):
                base_cfg = {
                    "selector_method": sel,
                    "selector_params": p_var,
                    "ar_lags": lags,
                    "agg_rule": agg,
                    "regression_model": reg_m,
                    "train_window": tw,
                    "step_size": sz,
                    "window_type": wt,
                    "mixed_frequency": args.mixed_frequency,
                    "seed": args.seed,
                }

                if wt == "rolling":
                    for rws in rolling_window_sizes:
                        cfg = dict(base_cfg)
                        cfg["rolling_window_size"] = rws
                        grids.append(cfg)
                else:
                    grids.append(base_cfg)

    if args.search_max_configs > 0 and len(grids) > args.search_max_configs:
        rng = random.Random(args.seed)
        rng.shuffle(grids)
        grids = grids[: args.search_max_configs]

    return grids


def main():
    args = build_experiment_parser().parse_args()

    logger.info("=== Bridge Equation Experiment Search Runner ===")
    logger.info(f"Seed: {args.seed}")

    X_panel, y_target = _load_bridge_data(args)
    logger.info(f"Panel shape: {X_panel.shape}, Target shape: {y_target.shape}")

    grid = generate_experiment_grid(args)
    logger.info(f"Generated {len(grid)} experiment configurations.")

    if not grid:
        logger.warning("Empty grid. Exiting.")
        return

    run_kwargs = {
        "X_panel": X_panel,
        "y_target": y_target,
        "search_mode": args.search_mode,
        "search_last_n_steps": args.search_last_n_steps,
    }

    logger.info(f"Running experiments (n_jobs={args.n_jobs}, mode={args.search_mode}) ...")
    tstart = time.time()

    if args.search_strategy == "staged":
        logger.info(f"--- Stage 1: Coarse Search ({len(grid)} configs, 3 steps) ---")
        stage1_kwargs = dict(run_kwargs)
        stage1_kwargs["search_last_n_steps"] = min(3, args.search_last_n_steps) if args.search_last_n_steps > 0 else 3

        if args.n_jobs > 1:
            results_s1 = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(run_single_experiment)(cfg, **stage1_kwargs) for cfg in grid
            )
        else:
            results_s1 = []
            for i, cfg in enumerate(grid, 1):
                logger.info(f"[{i}/{len(grid)}] Stage 1 Running: {cfg}")
                results_s1.append(run_single_experiment(cfg, **stage1_kwargs))

        df_s1 = pd.DataFrame([r for r in results_s1 if r.get("status") == "success"])
        if df_s1.empty:
            logger.error("Stage 1 failed completely.")
            return

        df_s1 = df_s1.sort_values("rmse", ascending=True).reset_index(drop=True)
        top_k = min(len(df_s1), args.search_top_k)
        best_configs = df_s1.head(top_k).to_dict("records")

        logger.info(f"--- Stage 2: Fine Search (Top {top_k} configs, {args.search_last_n_steps} steps) ---")
        clean_keys = set(grid[0].keys())
        staged_grid = [{k: v for k, v in cfg.items() if k in clean_keys} for cfg in best_configs]

        if args.n_jobs > 1:
            results = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(run_single_experiment)(cfg, **run_kwargs) for cfg in staged_grid
            )
        else:
            results = []
            for i, cfg in enumerate(staged_grid, 1):
                logger.info(f"[{i}/{len(staged_grid)}] Stage 2 Running: {cfg}")
                results.append(run_single_experiment(cfg, **run_kwargs))
    else:
        if args.n_jobs > 1:
            results = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(run_single_experiment)(cfg, **run_kwargs) for cfg in grid
            )
        else:
            results = []
            for i, cfg in enumerate(grid, 1):
                logger.info(f"[{i}/{len(grid)}] Running: {cfg}")
                results.append(run_single_experiment(cfg, **run_kwargs))

    total_time = time.time() - tstart
    df_res = pd.DataFrame(results)

    if df_res.empty:
        logger.error("No results produced.")
        return

    if "selector_params" in df_res.columns:
        df_res["selector_params"] = df_res["selector_params"].apply(json.dumps)

    success = df_res[df_res["status"] == "success"].copy()
    failed_count = len(df_res) - len(success)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if success.empty:
        logger.error("ALL experiments failed. Check logs.")
        if "_eval_df" in df_res.columns:
            df_res = df_res.drop(columns=["_eval_df"])
        df_res.to_csv(out_path, index=False)
        return

    success = success.sort_values("rmse", ascending=True).reset_index(drop=True)

    # Keep full output: successful runs first, then failed runs
    failures = df_res[df_res["status"] != "success"].copy()
    failures = failures.sort_values(by=["status"]).reset_index(drop=True)
    df_out = pd.concat([success, failures], ignore_index=True)

    logger.info(f"\nExperiment complete in {total_time / 60:.1f} minutes.")
    logger.info(f"Success: {len(success)}, Failed: {failed_count}")
    logger.info("\n--- Top 5 Configurations by RMSE ---")

    print(
        success.head(5)[
            [
                "selector_method",
                "selector_params",
                "ar_lags",
                "agg_rule",
                "regression_model",
                "window_type",
                "rmse",
                "mae",
                "runtime_sec",
            ]
        ].to_string(index=False)
    )

    best_eval_df = success.iloc[0]["_eval_df"]
    preds_path = out_path.parent / f"predictions_bridge_{args.seed}.csv"
    best_eval_df.to_csv(preds_path, index=False)
    logger.info(f"Bridge best preds → {preds_path}")

    if "_eval_df" in df_out.columns:
        df_out = df_out.drop(columns=["_eval_df"])

    df_out.to_csv(out_path, index=False)
    logger.info(f"\nDetailed results saved to {out_path.absolute()}")

    best_config = success.iloc[0].to_dict()
    best_config.pop("_eval_df", None)

    # selector_params is now JSON text in success
    if isinstance(best_config.get("selector_params"), str):
        try:
            best_config["selector_params"] = json.loads(best_config["selector_params"])
        except Exception:
            pass

    best_cfg_path = out_path.parent / "best_bridge_configuration.json"
    with open(best_cfg_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=4, default=str)
    logger.info(f"Best configuration saved to {best_cfg_path}")


if __name__ == "__main__":
    main()