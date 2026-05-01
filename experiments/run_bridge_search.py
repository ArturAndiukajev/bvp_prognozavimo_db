"""
Bridge Equation hiperparametrų paieškos ir požymių atrankos eksperimentų vykdymo
pipeline'as. Vykdo grid search per Bridge equation hiperparametrus, regresijos modelius
ir požymių atrankos strategijas. Išoriniai ciklai paralelizuojami,
kad būtų paspartintas backtest'inimas. Gauti modelių deriniai
surūšiuojami pagal RMSE/MAE ir eksportuojami į CSV failą. Taip pat yra geriausių
parametrų paieška su Optuna.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import time
from typing import Any, Dict, List
import pandas as pd
import optuna

# Force single-thread linear algebra to prevent collision in parallel workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Ensure project root is importable when run as a script directly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nowcasting.evaluation.checkpoint import prune_grid, run_chunks, build_run_signature
from nowcasting.utils.data_loader import load_mf_panels, load_cf_panel, _DEFAULT_DATA_DIR
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

    # ---------- Optuna ----------
    ap.add_argument("--engine", choices=["optuna", "grid", "staged", "evaluate"], default=None,
                    help="Search engine to use: optuna, grid, staged, or evaluate (single run from config)")
    ap.add_argument("--config-path", type=str, default=None, help="Path to a JSON configuration file for '--engine evaluate'")
    ap.add_argument("--n-trials", type=int, default=50, help="Number of trials for Optuna engine")
    ap.add_argument("--study-storage", type=str, default=None, 
                    help="Optuna storage URI. Defaults to sqlite in results directory.")
    ap.add_argument("--study-name", type=str, default=None, help="Optuna study name for persistence and resuming.")
    ap.add_argument("--save-every-n-trials", type=int, default=5, help="Save summary CSV every N trials")

    ap.add_argument("--n-jobs", type=int, default=1)

    ap.add_argument("--resume", action="store_true", help="Resume from an interrupted search using existing results CSV.")
    ap.add_argument("--checkpoint-chunk-size", type=int, default=10, help="Save to CSV after completing this many configurations.")
    ap.add_argument("--out-file", type=str, default="data/forecasts/bridge_search_results.csv")
    return ap


def parse_list(val_str: str, dtype=str):
    return [dtype(v.strip()) for v in val_str.split(",") if v.strip()]

def filter_feasible_train_windows(train_windows: List[int], n_total: int, min_test_steps: int = 5) -> List[int]:
    """Filters train windows that are too large for the dataset."""
    max_allowed = n_total - min_test_steps
    feasible = [tw for tw in train_windows if tw < max_allowed]
    
    if not feasible:
        # Fallback: generate windows as fractions of dataset length
        if n_total > 15:
            fallback = [int(n_total * 0.3), int(n_total * 0.5), int(n_total * 0.7)]
        else:
            fallback = [max(5, n_total - min_test_steps - 1)]
        feasible = sorted(list(set(fallback)))
        logger.info(f"Small dataset (n={n_total}): all requested windows were too large. Falling back to {feasible}")
    else:
        if len(feasible) < len(train_windows):
            logger.info(f"Small dataset (n={n_total}): restricted train_windows to {feasible} from original set.")
            
    return feasible


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
        selector_params["alpha"] = trial.suggest_float("elasticnet_selector_alpha", 1e-4, 1.0, log=True)
        selector_params["l1_ratio"] = trial.suggest_float("elasticnet_selector_l1", 0.1, 0.9)
    elif selector_name == "fast_screen":
        selector_params["top_n"] = trial.suggest_int("fast_screen_k", 20, 100)
    elif selector_name == "pca":
        selector_params["n_components"] = trial.suggest_int("pca_comp", 2, 20)
    elif selector_name == "factor_analysis":
        selector_params["n_components"] = trial.suggest_int("fa_comp", 2, 20)
    elif selector_name == "autoencoder":
        selector_params["latent_dim"] = trial.suggest_int("ae_dim", 2, 20)

    ar_lags = trial.suggest_int("ar_lags", 0, 12)
    agg_rule = trial.suggest_categorical("agg_rule", ["mean", "last", "sum"])
    reg_model = trial.suggest_categorical("regression_model", ["linear", "ridge", "lasso", "elasticnet"])

    # Use the pre-filtered windows from args if available, otherwise fallback to defaults
    tw_candidates = getattr(args, "feasible_train_windows", [60, 80, 100, 120])
    train_window = trial.suggest_categorical("train_window", tw_candidates)
    step_size = trial.suggest_categorical("step_size", [1, 3])
    window_type = trial.suggest_categorical("window_type", ["expanding", "rolling"])
    
    config = {
        "selector_method": selector_name,
        "selector_params": selector_params,
        "ar_lags": ar_lags,
        "agg_rule": agg_rule,
        "regression_model": reg_model,
        "train_window": train_window,
        "step_size": step_size,
        "window_type": window_type,
        "mixed_frequency": args.mixed_frequency,
        "seed": args.seed,
    }
    
    if window_type == "rolling":
        config["rolling_window_size"] = trial.suggest_categorical("rolling_window_size", [40, 60, 80])
        
    res = run_single_experiment(
        config=config,
        X_panel=X_panel,
        y_target=y_target,
        search_mode=args.search_mode,
        search_last_n_steps=args.search_last_n_steps,
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
            preds_path = tracker.out_dir / f"bridge_optuna_best_preds_{tracker.suffix}.csv"
            res["_eval_df"].to_csv(preds_path, index=False)
            logger.info(f"New best RMSE ({rmse:.4f}) -> {preds_path}")
            
    return rmse


def run_optuna_engine(X_panel: pd.DataFrame, y_target: pd.Series, args: argparse.Namespace, out_dir: Path, suffix: str):
    logger.info("Starting OPTUNA engine.")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    study_name = args.study_name or f"bridge_{suffix}"
    storage_url = args.study_storage or f"sqlite:///{out_dir.absolute()}/bridge_optuna.db"
    
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
        if k not in ["rmse", "mae", "status", "runtime_sec", "n_eval_points", "n_folds", "error_message"]:
            eval_config[k] = _parse_if_str(v)

    logger.info(f"Evaluating Bridge with config from {args.config_path}")

    search_grid = [eval_config]
    
    run_signature = build_run_signature(
        script_name="run_bridge_search.py",
        input_panel=str(args.input_panel or args.data_dir),
        target=str(args.target),
        seed=args.seed,
        search_last_n_steps=args.search_last_n_steps,
        search_strategy=args.search_strategy,
        mixed_frequency=args.mixed_frequency,
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
        "search_last_n_steps": args.search_last_n_steps
    }

    results = run_chunks(pruned_grid, run_single_experiment, run_kwargs, out_path, n_jobs=1, chunk_sz=1, stage_name="Evaluation")
    
    if results and results[0]["status"] == "success":
        res = results[0]
        logger.info(f"Evaluation Success | RMSE: {res['rmse']:.4f} | MAE: {res['mae']:.4f}")
        eval_df = res.pop("_eval_df", pd.DataFrame())
        if not eval_df.empty:
            out_preds = out_dir / f"bridge_eval_preds_{suffix}.csv"
            eval_df.to_csv(out_preds, index=False)
            logger.info(f"Saved predictions to {out_preds}")
        
        out_json = out_dir / f"bridge_eval_results_{suffix}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in res.items() if k != "_eval_df"}, f, indent=4, default=str)
        logger.info(f"Saved results to {out_json}")
    else:
        err = results[0].get("error_message") if results else "Unknown error"
        logger.error(f"Evaluation failed: {err}")


def generate_experiment_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    selectors_to_run = parse_list(args.selectors, str)
    ar_lags_list = parse_list(args.ar_lags, int)
    agg_rules = parse_list(args.agg_rules, str)
    regression_models = parse_list(args.regression_models, str)
    train_windows = getattr(args, "feasible_train_windows", parse_list(args.train_windows, int))
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

    # 1.5 Filter feasible windows
    requested_windows = parse_list(args.train_windows, int)
    args.feasible_train_windows = filter_feasible_train_windows(requested_windows, len(y_target.dropna()))

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
        logger.info(f"--- Stage 1: Coarse Search ({len(grid)} configs, 3 steps) ---")
        stage1_kwargs = dict(run_kwargs)
        stage1_steps = min(3, args.search_last_n_steps) if args.search_last_n_steps > 0 else 3
        stage1_kwargs["search_last_n_steps"] = stage1_steps
        
        run_signature = build_run_signature(
            script_name="run_bridge_search.py",
            input_panel=str(args.input_panel or args.data_dir),
            target=str(args.target),
            seed=args.seed,
            search_last_n_steps=args.search_last_n_steps,
            search_strategy=args.search_strategy,
            mixed_frequency=args.mixed_frequency
        )
        
        grid_s1 = prune_grid(grid, stage1_steps, out_path, getattr(args, "resume", False), run_signature=run_signature)
        results_s1 = run_chunks(grid_s1, run_single_experiment, stage1_kwargs, out_path, n_jobs=args.n_jobs, chunk_sz=getattr(args, "checkpoint_chunk_size", 10), stage_name="Stage 1")

        if out_path.exists():
            try:
                full_s1_df = pd.read_csv(out_path)
                full_s1_df = full_s1_df[(full_s1_df["status"] == "success")]
                best_configs = full_s1_df.sort_values("rmse").head(args.search_top_k).to_dict("records")
            except Exception:
                df_s1 = pd.DataFrame([r for r in results_s1 if r.get("status") == "success"])
                best_configs = df_s1.sort_values("rmse", ascending=True).head(args.search_top_k).to_dict("records") if not df_s1.empty else []
        else:
            df_s1 = pd.DataFrame([r for r in results_s1 if r.get("status") == "success"])
            best_configs = df_s1.sort_values("rmse", ascending=True).head(args.search_top_k).to_dict("records") if not df_s1.empty else []

        if not best_configs:
            logger.error("Stage 1 failed completely.")
            return

        logger.info(f"--- Stage 2: Fine Search (Top {len(best_configs)} configs, {args.search_last_n_steps} steps) ---")
        clean_keys = set(grid[0].keys())
        staged_grid = [{k: v for k, v in cfg.items() if k in clean_keys} for cfg in best_configs]
        staged_grid = prune_grid(staged_grid, args.search_last_n_steps, out_path, getattr(args, "resume", False), run_signature=run_signature)

        results = run_chunks(staged_grid, run_single_experiment, run_kwargs, out_path, n_jobs=args.n_jobs, chunk_sz=getattr(args, "checkpoint_chunk_size", 10), stage_name="Stage 2")
    else:
        run_signature = build_run_signature(
            script_name="run_bridge_search.py",
            input_panel=str(args.input_panel or args.data_dir),
            target=str(args.target),
            seed=args.seed,
            search_last_n_steps=args.search_last_n_steps,
            search_strategy=args.search_strategy,
            mixed_frequency=args.mixed_frequency
        )
        grid = prune_grid(grid, args.search_last_n_steps, out_path, getattr(args, "resume", False), run_signature=run_signature)
        results = run_chunks(grid, run_single_experiment, run_kwargs, out_path, n_jobs=args.n_jobs, chunk_sz=getattr(args, "checkpoint_chunk_size", 10))

    total_time = time.time() - tstart

    if out_path.exists():
        df_res = pd.read_csv(out_path)
    else:
        df_res = pd.DataFrame(results)

    if df_res.empty:
        logger.error("No results produced.")
        return

    if "selector_params" in df_res.columns:
        df_res["selector_params"] = df_res["selector_params"].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)

    success = df_res[df_res["status"] == "success"].copy()
    failed_count = len(df_res) - len(success)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if success.empty:
        logger.error("ALL experiments failed. Check logs.")
        if "_eval_df" in df_res.columns:
            df_res = df_res.drop(columns=["_eval_df"])
        df_res.to_csv(out_path, index=False)
        return

    success = success.sort_values("rmse", ascending=True).reset_index(drop=True)

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

    best_cfg = success.iloc[0].to_dict()
    memory_res = [r for r in results if r.get("_config_id") == best_cfg.get("_config_id")]
    if memory_res and "_eval_df" in memory_res[0]:
        best_eval_df = memory_res[0]["_eval_df"]
    else:
        logger.info("Re-evaluating best configuration to regenerate predictions output...")
        b_cfg = {k:v for k,v in best_cfg.items() if not str(k).startswith("_")}
        if isinstance(b_cfg.get('selector_params'), str):
            b_cfg['selector_params'] = json.loads(b_cfg['selector_params'])
        b_res = run_single_experiment(b_cfg, **run_kwargs)
        best_eval_df = b_res.get("_eval_df", pd.DataFrame())

    if not best_eval_df.empty:
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