"""
run_tactis_search.py  —  TACTiS Nowcasting Grid-Search Experiment Runner
=========================================================================

Searches over practical TACTiS hyperparameters:
  - Feature selectors (none, corr_top_n, variance_filter, lasso, elasticnet)
  - Training params  (history_length, epochs, batch_size, lr, num_samples)
  - Architecture     (skip_copula)
  - Backtest params  (train_window, step_size)

Key design guarantees
---------------------
  ✓ No look-ahead bias: selectors fit only on the training fold
  ✓ Failed configs stored; search continues
  ✓ Seed set per run for reproducibility
  ✓ Quick-search mode available via --search-mode quick

Warning: TACTiS is computationally expensive. Start with a very small grid
(--search-mode quick) and scale up as budget allows. GPU is highly recommended.

Quick start
-----------
    python scripts/run_tactis_search.py \\
        --selectors none,corr_top_n \\
        --top-n 10 \\
        --history-lengths 12 \\
        --epochs-list 3 \\
        --batch-sizes 16 \\
        --num-samples-list 20 \\
        --skip-copula-options true \\
        --train-windows 80 \\
        --search-last-n-steps 5 \\
        --n-jobs 1 \\
        --seed 42

Full search
-----------
    python scripts/run_tactis_search.py \\
        --selectors none,corr_top_n,variance_filter,lasso,elasticnet \\
        --top-n 10,20 \\
        --variance-thresholds 1e-6 \\
        --lasso-alphas 0.001,0.01 \\
        --elasticnet-alphas 0.001 \\
        --elasticnet-l1-ratios 0.5 \\
        --history-lengths 12,24 \\
        --epochs-list 5,10 \\
        --batch-sizes 16,32 \\
        --learning-rates 1e-4,1e-3 \\
        --num-samples-list 20,50 \\
        --skip-copula-options false,true \\
        --train-windows 60,80 \\
        --step-sizes 1,3 \\
        --search-last-n-steps 20 \\
        --n-jobs 1 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Single-thread BLAS to avoid nested-parallelism collisions
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nowcasting.main import load_cf_panel, _DEFAULT_DATA_DIR
from nowcasting.evaluation.backtester import RollingBacktester
from nowcasting.evaluation.metrics import compute_metrics
from nowcasting.models.tactis_wrapper import TACTiSNowcastWrapper

from nowcasting.features.selectors import (
    IdentitySelector,
    VarianceFilter,
    CorrTopNSelector,
    LassoSelector,
    ElasticNetSelector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tactis_search")


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def _sanitize(s: str) -> str:
    return re.sub(r"[^\w\-.]", "_", str(s))


def _shorten(s: str, max_len: int = 40) -> str:
    s = str(s)
    if len(s) <= max_len:
        return s
    k = max_len // 2 - 1
    return s[:k] + "_" + s[-k:]


def _build_suffix(panel_path: Optional[Path], seed: int) -> str:
    stem = _shorten(_sanitize(panel_path.stem if panel_path else "panel"), 30)
    return f"tactis_{stem}_s{seed}"


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="TACTiS Nowcasting Grid-Search Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    ap.add_argument("--data-dir", default=_DEFAULT_DATA_DIR)
    ap.add_argument("--input-panel", default=None)
    ap.add_argument("--target-col", default=None, dest="target")

    # Feature selectors
    ap.add_argument("--selectors", default="none,corr_top_n",
                    help="none, variance_filter, corr_top_n, lasso, elasticnet")
    ap.add_argument("--variance-thresholds", default="1e-6")
    ap.add_argument("--top-n", default="10,20")
    ap.add_argument("--lasso-alphas", default="0.001,0.01")
    ap.add_argument("--elasticnet-alphas", default="0.001,0.01")
    ap.add_argument("--elasticnet-l1-ratios", default="0.5")

    # TACTiS training params
    ap.add_argument("--history-lengths", default="12,24")
    ap.add_argument("--epochs-list", default="5,10")
    ap.add_argument("--batch-sizes", default="16,32")
    ap.add_argument("--learning-rates", default="1e-4,1e-3")
    ap.add_argument("--num-samples-list", default="20,50")
    ap.add_argument("--skip-copula-options", default="false,true",
                    help="'false' = full TACTiS-2, 'true' = flow-only (faster)")

    # Backtest
    ap.add_argument("--train-windows", default="80,120")
    ap.add_argument("--step-sizes", default="1")
    ap.add_argument("--search-last-n-steps", type=int, default=0)
    ap.add_argument("--search-max-configs", type=int, default=0)

    # Search mode
    ap.add_argument("--search-mode", choices=["quick", "full"], default="full",
                    help="quick: cap epochs<=5, num_samples<=20; full: use given values")

    # Compute
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Parallel workers. WARNING: TACTiS is expensive; use n_jobs=1 unless "
                         "you have many CPU cores or GPUs.")
    ap.add_argument("--seed", type=int, default=42)

    # Output
    ap.add_argument("--results-file", default="data/forecasts/tactis_experiment_results.csv")

    return ap


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def _plist(s: str, dtype=str) -> list:
    return [dtype(v.strip()) for v in s.split(",") if v.strip()]


def _parse_bool(s: str) -> bool:
    return s.strip().lower() in ("true", "1", "yes")


def _selector_param_grid(method: str, args: argparse.Namespace) -> List[dict]:
    if method == "none":
        return [{}]
    elif method == "variance_filter":
        return [{"threshold": t} for t in _plist(args.variance_thresholds, float)]
    elif method == "corr_top_n":
        return [{"top_n": n} for n in _plist(args.top_n, int)]
    elif method == "lasso":
        return [{"alpha": a} for a in _plist(args.lasso_alphas, float)]
    elif method == "elasticnet":
        return [
            {"alpha": a, "l1_ratio": l}
            for a in _plist(args.elasticnet_alphas, float)
            for l in _plist(args.elasticnet_l1_ratios, float)
        ]
    return [{}]


def build_selector(method: str, params: dict) -> Any:
    if method == "none":
        return IdentitySelector()
    elif method == "variance_filter":
        return VarianceFilter(threshold=params.get("threshold", 1e-6))
    elif method == "corr_top_n":
        return CorrTopNSelector(top_n=params.get("top_n", 20))
    elif method == "lasso":
        return LassoSelector(alpha=params.get("alpha", 0.1))
    elif method == "elasticnet":
        return ElasticNetSelector(alpha=params.get("alpha", 0.1),
                                  l1_ratio=params.get("l1_ratio", 0.5))
    raise ValueError(f"Unknown selector: {method}")


def _tactis_param_grid(args: argparse.Namespace) -> List[dict]:
    skip_copula_vals = [_parse_bool(v) for v in _plist(args.skip_copula_options, str)]
    return [
        {
            "history_length": hl,
            "epochs":         ep,
            "batch_size":     bs,
            "lr":             lr,
            "num_samples":    ns,
            "skip_copula":    sc,
        }
        for hl in _plist(args.history_lengths, int)
        for ep in _plist(args.epochs_list, int)
        for bs in _plist(args.batch_sizes, int)
        for lr in _plist(args.learning_rates, float)
        for ns in _plist(args.num_samples_list, int)
        for sc in skip_copula_vals
    ]


def generate_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    selectors_to_run = _plist(args.selectors, str)
    train_windows    = _plist(args.train_windows, int)
    step_sizes       = _plist(args.step_sizes, int)
    tactis_grid      = _tactis_param_grid(args)

    # Apply quick-mode caps
    if args.search_mode == "quick":
        tactis_grid = [
            {**p, "epochs": min(p["epochs"], 5), "num_samples": min(p["num_samples"], 20)}
            for p in tactis_grid
        ]
        # Deduplicate after capping
        seen, unique_grid = set(), []
        for p in tactis_grid:
            key = json.dumps(p, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_grid.append(p)
        tactis_grid = unique_grid
        logger.info(f"Quick mode: {len(tactis_grid)} unique TACTiS param sets after capping.")

    configs: List[Dict[str, Any]] = []
    for sel_method in selectors_to_run:
        for sel_params in _selector_param_grid(sel_method, args):
            for tactis_p, tw, sz in itertools.product(tactis_grid, train_windows, step_sizes):
                configs.append({
                    "selector_method": sel_method,
                    "selector_params": sel_params,
                    "tactis_params":   tactis_p,
                    "train_window":    tw,
                    "step_size":       sz,
                })

    if args.search_max_configs > 0 and len(configs) > args.search_max_configs:
        random.shuffle(configs)
        configs = configs[:args.search_max_configs]

    return configs


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_single_experiment(
    config: Dict[str, Any],
    X_panel: pd.DataFrame,
    y_target: pd.Series,
    search_last_n_steps: int,
    seed: int,
) -> dict:
    """Run one TACTiS configuration. Never raises; failed configs are recorded."""
    np.random.seed(seed)

    tp = config["tactis_params"]
    res: dict = {
        "model":             "TACTiS",
        "selector_method":   config["selector_method"],
        "selector_params":   json.dumps(config["selector_params"]),
        "tactis_params":     json.dumps(tp),
        "train_window":      config["train_window"],
        "step_size":         config["step_size"],
        "n_features_input":  X_panel.shape[1],
        "n_features_used":   None,
        "rmse":              None,
        "mae":               None,
        "mape":              None,
        "n_eval_points":     None,
        "runtime_sec":       None,
        "seed":              seed,
        "history_length":    tp["history_length"],
        "epochs":            tp["epochs"],
        "batch_size":        tp["batch_size"],
        "lr":                tp["lr"],
        "num_samples":       tp["num_samples"],
        "skip_copula":       tp["skip_copula"],
        "window_type":       "expanding",
        "eval_mode":         "common_frequency",
        "status":            "running",
        "error_message":     "",
    }

    t0 = time.time()
    try:
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

        selector = build_selector(config["selector_method"], config["selector_params"])

        model = TACTiSNowcastWrapper(
            target_col=str(y_target.name),
            seed=seed,
            **tp,
        )

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
            res["status"]        = "failed"
            res["error_message"] = "Backtester returned empty DataFrame"
            return res

        metrics = compute_metrics(eval_df, model_name="TACTiS")
        res["rmse"]         = float(metrics["RMSE"].iloc[0])
        res["mae"]          = float(metrics["MAE"].iloc[0])
        res["mape"]         = float(metrics["MAPE"].iloc[0]) if "MAPE" in metrics.columns else float("nan")
        res["n_eval_points"]= len(eval_df)
        res["n_folds"]      = eval_df["Forecast_Origin"].nunique() if "Forecast_Origin" in eval_df.columns else len(eval_df)

        if hasattr(selector, "selected_cols_") and selector.selected_cols_ is not None:
            res["n_features_used"] = len(selector.selected_cols_)
        else:
            res["n_features_used"] = X_panel.shape[1]

        res["status"] = "success"

    except Exception as exc:
        res["status"]       = "failed"
        res["error_message"] = str(exc)
        res["runtime_sec"]  = round(time.time() - t0, 2)
        logger.debug(f"Config failed [TACTiS]: {exc}", exc_info=True)

    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap   = build_parser()
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("  TACTiS Nowcasting Grid-Search Experiment Runner")
    logger.info(f"  seed={args.seed}  |  n_jobs={args.n_jobs}  |  mode={args.search_mode}")
    logger.info("=" * 60)

    if args.n_jobs > 1:
        logger.warning(
            f"Parallel mode with n_jobs={args.n_jobs}. TACTiS is memory-heavy — "
            "ensure you have sufficient RAM/VRAM. Consider n_jobs=1 on CPU-only machines."
        )

    # 1. Load panel
    data_dir = Path(args.data_dir)
    logger.info("Loading panel ...")
    X_panel, y_target = load_cf_panel(data_dir, args.target, panel_arg=args.input_panel)
    logger.info(f"Panel: {X_panel.shape}  |  Target: {y_target.name}")

    # Resolve panel stem for filenames
    if args.input_panel:
        _ip = Path(args.input_panel)
        panel_path: Optional[Path] = _ip if _ip.exists() else Path(args.data_dir) / _ip
    else:
        panel_path = None
    dataset_stem = panel_path.stem if panel_path else "panel"

    # 2. Build grid
    grid = generate_grid(args)
    logger.info(f"Generated {len(grid)} experiment configurations.")
    if not grid:
        logger.warning("Empty grid — check CLI arguments. Exiting.")
        return

    # 3. Run
    logger.info(f"Running experiments  (parallel={args.n_jobs > 1}, workers={args.n_jobs})")
    tstart = time.time()

    run_kwargs = dict(
        X_panel=X_panel,
        y_target=y_target,
        search_last_n_steps=args.search_last_n_steps,
        seed=args.seed,
    )

    if args.n_jobs > 1:
        results = Parallel(n_jobs=args.n_jobs, verbose=5, prefer="processes")(
            delayed(run_single_experiment)(cfg, **run_kwargs) for cfg in grid
        )
    else:
        results = []
        for i, cfg in enumerate(grid, 1):
            tp = cfg["tactis_params"]
            logger.info(
                f"[{i}/{len(grid)}]  "
                f"sel={cfg['selector_method']:<16s}  "
                f"hl={tp['history_length']}  ep={tp['epochs']}  "
                f"skip_copula={tp['skip_copula']}  tw={cfg['train_window']}"
            )
            results.append(run_single_experiment(cfg, **run_kwargs))

    total_time = time.time() - tstart

    # 4. Collect + sort
    df_res = pd.DataFrame(results)
    df_res["dataset_stem"] = dataset_stem
    df_res["dataset_file"] = str(panel_path or args.input_panel or args.data_dir)

    success      = df_res[df_res["status"] == "success"]
    failed_count = len(df_res) - len(success)

    logger.info(
        f"\nFinished in {total_time / 60:.1f} min  |  "
        f"Success: {len(success)}  |  Failed: {failed_count}"
    )

    out_base = Path(args.results_file)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    if not success.empty:
        df_res = df_res.sort_values(["rmse", "mae"], ascending=True).reset_index(drop=True)

        display_cols = [c for c in [
            "selector_method", "history_length", "epochs", "skip_copula",
            "train_window", "rmse", "mae", "runtime_sec"
        ] if c in df_res.columns]
        logger.info("\n--- Top 5 Configurations by RMSE ---")
        print(df_res.head(5)[display_cols].to_string(index=False))

    # 5. Save results
    suffix    = _build_suffix(panel_path, args.seed)
    csv_path  = out_base.parent / f"gridsearch_{suffix}.csv"
    df_res.to_csv(csv_path, index=False)
    logger.info(f"Results              → {csv_path}")

    # Also write combined table at specified path
    df_res.to_csv(out_base, index=False)
    logger.info(f"Combined results     → {out_base.absolute()}")

    if not success.empty:
        best = success.sort_values("rmse").iloc[0].to_dict()
        for jcol in ("selector_params", "tactis_params"):
            try:
                best[jcol] = json.loads(best[jcol])
            except Exception:
                pass
        best_path = out_base.parent / f"best_config_{suffix}.json"
        with open(best_path, "w") as fh:
            json.dump(best, fh, indent=4, default=str)
        logger.info(f"Best config         → {best_path}")
    else:
        logger.error("ALL experiments failed — check error_message column in results CSV.")


if __name__ == "__main__":
    main()
