"""
nowcasting/main.py
==================
Nowcasting framework entry-point.

Two modes
---------
common_frequency  — loads panel_monthly.parquet, runs ElasticNet / LightGBM /
                    PCA regression / DFM (CF) / TACTiS-2.
mixed_frequency   — loads mf_panel_M.parquet + mf_panel_Q.parquet, runs
                    MIDAS / Bridge Equation / DFM (MQ).

Usage
-----
# Common-frequency pipeline
python -m nowcasting.main --mode common_frequency --target <series_id>

# Mixed-frequency pipeline
python -m nowcasting.main --mode mixed_frequency --target <series_id> \\
       --lf-freq QE --freq-ratio 3
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd

os.environ.setdefault("POLARS_MAX_THREADS", "8")

from nowcasting.utils.data_loader import (
    load_cf_panel,
    load_mf_panels,
    validate_inputs,
    _DEFAULT_DATA_DIR,
    _DEFAULT_OUT_DIR,
)

from nowcasting.features.engineering import TimeSeriesFeatureEngineer
from nowcasting.features.selection import MultiStageFeatureSelector
from nowcasting.evaluation.backtester import RollingBacktester
from nowcasting.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("nowcast_main")





# ---------------------------------------------------------------------------
# Common-Frequency Pipeline
# ---------------------------------------------------------------------------

def run_common_frequency(args: argparse.Namespace) -> None:
    logger.info("=== Common-Frequency Nowcasting Pipeline ===")

    data_dir = Path(args.data_dir)
    X_panel, y_target = load_cf_panel(data_dir, args.target, panel_arg=args.input_panel)

    if not validate_inputs(X_panel, y_target, "CF"):
        logger.error("Input validation failed -- aborting.")
        return

    target_name = str(y_target.name)

    from nowcasting.models.ml_regression import ElasticNetNowcast, LightGBMNowcast
    from nowcasting.models.pca_regression import PCARegressionNowcast
    from nowcasting.models.dfm import DynamicFactorNowcast
    from nowcasting.models.bvar import BVARNowcast
    from nowcasting.models.midas import MIDASNowcast
    from nowcasting.models.bridge_equation import BridgeEquationNowcast

    # --- Core regression models (always on) ---
    models = {
        "ElasticNet":     ElasticNetNowcast(target_col=target_name),
        "LightGBM":       LightGBMNowcast(target_col=target_name),
        "PCA_Regression": PCARegressionNowcast(
            target_col=target_name,
            n_components=min(args.pca_components, X_panel.shape[1]),
        ),
    }

    # --- State-space models ---
    if not args.no_dfm:
        models["DFM"] = DynamicFactorNowcast(
            target_col=target_name,
            k_factors=args.k_factors,
            mixed_frequency=False,
        )

    if not args.no_bvar:
        models["BVAR"] = BVARNowcast(
            target_col=target_name,
            maxlags=args.bvar_lags,
        )

    # --- Deep learning: VAE ---
    if not args.no_vae:
        try:
            import torch  # noqa: F401
            models["VAE"] = VAENowcast(
                target_col=target_name,
                latent_dim=args.vae_latent_dim,
                hidden_dim=args.vae_hidden_dim,
                epochs=args.vae_epochs,
            )
        except ImportError:
            logger.warning("PyTorch not installed -- skipping VAE model.")

    # --- Deep learning: TACTiS-2 ---
    if not args.no_tactis:
        try:
            from nowcasting.models.tactis_wrapper import TACTiSNowcastWrapper
            models["TACTiS2"] = TACTiSNowcastWrapper(
                target_col=target_name,
                history_length=24,
                epochs=args.tactis_epochs,
            )
        except (ImportError, Exception) as e:
            logger.warning(f"TACTiS not available -- skipping TACTiS2. ({e})")

    # --- Mixed-frequency econometric models in CF mode ---
    # freq_ratio=1 => MIDAS degrades to a standard lag regression; Bridge runs AR(p) on same-freq X
    if not args.no_midas:
        models["MIDAS"] = MIDASNowcast(
            target_col=target_name,
            freq_ratio=1,          # same frequency in CF mode
            n_lags=args.midas_lags,
        )

    if not args.no_bridge:
        models["BridgeEquation"] = BridgeEquationNowcast(
            target_col=target_name,
            freq_ratio=1,
            ar_lags=args.bridge_ar_lags,
            lf_freq=args.lf_freq,
        )

    logger.info(f"Models to backtest ({len(models)}): {list(models.keys())}")

    initial_train = min(args.initial_train, int(len(X_panel) * 0.6))
    backtester = RollingBacktester(initial_train_periods=initial_train, step_size=1)

    selector = MultiStageFeatureSelector(
        method="pca",
        n_components=min(args.pca_components, X_panel.shape[1]),
    )

    _run_backtest_loop(models, X_panel, y_target, backtester, selector, args)


# ---------------------------------------------------------------------------
# Mixed-Frequency Pipeline
# ---------------------------------------------------------------------------

def run_mixed_frequency(args: argparse.Namespace) -> None:
    logger.info("=== Mixed-Frequency Nowcasting Pipeline ===")

    data_dir = Path(args.data_dir)
    X_m, X_q, y_target = load_mf_panels(data_dir, args.target, lf_freq=args.lf_freq, panel_arg=args.input_panel)

    # For MIDAS / Bridge: use monthly X, quarterly y
    if not validate_inputs(X_m if not X_m.empty else X_q, y_target, "MF"):
        logger.error("Input validation failed — aborting.")
        return

    target_name = str(y_target.name) if y_target.name else "target"
    freq_ratio = args.freq_ratio  # e.g. 3 for monthly→quarterly

    from nowcasting.models.midas import MIDASNowcast
    from nowcasting.models.bridge_equation import BridgeEquationNowcast

    models = {
        "MIDAS": MIDASNowcast(
            target_col=target_name,
            freq_ratio=freq_ratio,
            n_lags=args.midas_lags,
        ),
        "BridgeEquation": BridgeEquationNowcast(
            target_col=target_name,
            freq_ratio=freq_ratio,
            ar_lags=args.bridge_ar_lags,
            lf_freq=args.lf_freq,
        ),
    }

    if args.enable_dfm:
        q_cols = list(X_q.columns) if not X_q.empty else []
        # Combine monthly + quarterly for the MQ-DFM
        X_combined = pd.concat([X_m, X_q], axis=1) if not X_q.empty else X_m
        
        from nowcasting.models.dfm import DynamicFactorNowcast
        models["DFM_MQ"] = DynamicFactorNowcast(
            target_col=target_name,
            k_factors=args.k_factors,
            mixed_frequency=True,
            quarterly_cols=q_cols,
        )

    # Mixed-frequency backtester uses X_m (monthly) as the high-freq predictor
    X_hf = X_m if not X_m.empty else X_q
    initial_train = min(args.initial_train, int(len(y_target) * 0.6))
    backtester = RollingBacktester(initial_train_periods=initial_train, step_size=1)

    _run_backtest_loop(models, X_hf, y_target, backtester, None, args)


# ---------------------------------------------------------------------------
# Shared backtest loop
# ---------------------------------------------------------------------------

def _run_backtest_loop(
    models: dict,
    X_panel: pd.DataFrame,
    y_target: pd.Series,
    backtester: RollingBacktester,
    selector,
    args: argparse.Namespace,
) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for name, model in models.items():
        logger.info(f"\n--- Backtesting Model: {name} ---")
        eval_df = backtester.backtest(
            model=model,
            X_panel=X_panel,
            y_target=y_target,
            transformer=None,
            feature_selector=selector,
        )

        if not eval_df.empty:
            metrics_df = compute_metrics(eval_df, model_name=name)
            all_metrics.append(metrics_df)

            out_path = out_dir / f"{name}_eval.csv"
            eval_df.to_csv(out_path)
            logger.info(f"Saved forecasts: {out_path}")

    if all_metrics:
        final_report = pd.concat(all_metrics).reset_index(drop=True)
        logger.info(f"\n=== Evaluation Report ===\n{final_report.to_string(index=False)}")
        report_path = out_dir / "evaluation_summary.csv"
        final_report.to_csv(report_path, index=False)
        logger.info(f"Saved evaluation summary: {report_path}")
    else:
        logger.warning("No model produced valid backtest results.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run Nowcasting Framework")

    # Mode
    ap.add_argument(
        "--mode",
        type=str,
        default="common_frequency",
        choices=["common_frequency", "mixed_frequency"],
        help="Pipeline mode: common_frequency or mixed_frequency",
    )

    # Data
    ap.add_argument("--data-dir", type=str, default=_DEFAULT_DATA_DIR,
                    help="Directory containing preprocessed parquet files (CWD-independent)")
    ap.add_argument("--out-dir", type=str, default=_DEFAULT_OUT_DIR,
                    help="Directory to write forecast CSVs (CWD-independent)")
    ap.add_argument("--target", type=str, default=None, help="Target column name/series_id to nowcast")
    ap.add_argument("--input-panel", type=str, default=None, 
                    help="Relative or absolute path to a specific parquet panel file. Overrides newest-file default auto-discovery.")
    ap.add_argument("--list-panels", action="store_true", help="Print available panel files in data-dir and exit")

    # Common-frequency options
    ap.add_argument("--initial-train", type=int, default=120,
                    help="Minimum number of periods for the initial training window")
    ap.add_argument("--pca-components", type=int, default=10,
                    help="Number of PCA components for dimensionality reduction")
    ap.add_argument("--no-dfm",    action="store_true", help="Disable DFM")
    ap.add_argument("--no-bvar",   action="store_true", help="Disable BVAR")
    ap.add_argument("--no-vae",    action="store_true", help="Disable VAE (needs PyTorch)")
    ap.add_argument("--no-tactis", action="store_true", help="Disable TACTiS-2")
    ap.add_argument("--no-midas",  action="store_true", help="Disable MIDAS")
    ap.add_argument("--no-bridge", action="store_true", help="Disable Bridge Equation")
    ap.add_argument("--k-factors",      type=int, default=2,  help="DFM latent factors")
    ap.add_argument("--bvar-lags",      type=int, default=3,  help="BVAR max lags")
    ap.add_argument("--vae-latent-dim", type=int, default=8,  help="VAE latent space size")
    ap.add_argument("--vae-hidden-dim", type=int, default=64, help="VAE hidden layer width")
    ap.add_argument("--vae-epochs",     type=int, default=30, help="VAE training epochs")
    ap.add_argument("--tactis-epochs",  type=int, default=5,  help="TACTiS-2 training epochs")

    # Mixed-frequency options
    ap.add_argument("--lf-freq", type=str, default="QE",
                    help="Low-frequency target resample alias (e.g. QE for quarterly end)")
    ap.add_argument("--freq-ratio", type=int, default=3,
                    help="High-to-low frequency ratio (e.g. 3 for monthly→quarterly)")
    ap.add_argument("--midas-lags",     type=int, default=2, help="U-MIDAS number of lags")
    ap.add_argument("--bridge-ar-lags", type=int, default=3, help="Bridge Equation AR lags")

    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    
    data_dir_path = Path(args.data_dir)

    if getattr(args, "list_panels", False):
        print(f"\nAvailable panel parquet files in {data_dir_path}:")
        if not data_dir_path.exists():
            print("  (Directory does not exist)")
        else:
            files = list(data_dir_path.glob("*.parquet"))
            if not files:
                print("  (No parquet files found)")
            for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
                mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                print(f"  {f.name}   [Modified: {mtime}]")
        print()
        return

    logger.info(f"=== Scalable Macroeconomic Nowcasting Framework | mode={args.mode} ===")

    if args.mode == "common_frequency":
        run_common_frequency(args)
    else:
        run_mixed_frequency(args)


if __name__ == "__main__":
    main()
