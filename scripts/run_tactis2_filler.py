import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from nowcasting.data.vintage_builder import VintageBuilder
from scripts.local_data_builder import LocalDataManager
from nowcasting.fillers.tactis2_filler import TACTiS2Filler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_tactis2")

def main():
    parser = argparse.ArgumentParser(description="Standalone TACTiS-2 Filler Script")
    parser.add_argument("--dataset", type=str, default="baseline_common", help="Dataset name")
    parser.add_argument("--target-quarter", type=str, default="2020Q1", help="Target quarter (e.g., 2020Q1)")
    parser.add_argument("--vintage", type=str, default="-2", help="Vintage label")
    parser.add_argument("--train-start", type=str, default="2000-01-01", help="Training start date")
    parser.add_argument("--monthly-feature-release-lag-months", type=int, default=1, help="Macro release lag")
    parser.add_argument("--gt-release-lag-months", type=int, default=0, help="Google Trends release lag")
    parser.add_argument("--seed", type=int, default=2234, help="Random seed")
    parser.add_argument("--max-epochs", type=int, default=1, help="Max epochs for TACTiS2")
    parser.add_argument("--context-length", type=int, default=120, help="Context length")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--device", type=str, default="auto", help="torch device")
    parser.add_argument("--force-refit", action="store_true", help="Force retraining TACTiS2")
    parser.add_argument("--debug-preselect-top-k", type=int, default=None, help="Keep only top K features by absolute correlation with visible target.")
    
    args = parser.parse_args()
    
    dm = LocalDataManager(_PROJECT_ROOT)
    X, y, info = dm.load_or_build_dataset(args.dataset, force_rebuild=False)
    
    vb = VintageBuilder(dataset_name=args.dataset, random_state=args.seed)
    target_q_end = pd.Period(args.target_quarter, freq='Q').end_time.normalize()
    cutoff_date = vb.get_cutoff_date(target_q_end, args.vintage)
    
    logger.info(f"Target: {args.target_quarter} | Vintage: {args.vintage} | Cutoff: {cutoff_date.date()}")
    
    # Simulate VintageBuilder truncation/masking logic for the filler input
    X_visible = X.copy()
    for col in X_visible.columns:
        if col.startswith("gt_") or "_gt" in col or col.startswith("GT_"):
            lag = args.gt_release_lag_months
        else:
            lag = args.monthly_feature_release_lag_months
            
        if lag > 0:
            max_obs_date = cutoff_date - pd.offsets.MonthEnd(lag)
            X_visible.loc[X_visible.index > max_obs_date, col] = pd.NA
            
    X_visible = X_visible[X_visible.index <= cutoff_date]
    X_visible = X_visible[X_visible.index >= pd.Timestamp(args.train_start)]
    
    # Point 5: Add feature preselection for heavy Neural Nets
    if args.debug_preselect_top_k is not None:
        logger.info(f"Pre-selecting top {args.debug_preselect_top_k} features...")
        y_visible = y.reindex(X_visible.index).ffill()
        if not y_visible.dropna().empty:
            corrs = X_visible.corrwith(y_visible.dropna())
            top_cols = corrs.abs().sort_values(ascending=False).head(args.debug_preselect_top_k).index.tolist()
            X_visible = X_visible[top_cols]
            logger.info(f"Reduced features to {len(X_visible.columns)}")
    
    # 2. Initialize Filler
    filler = TACTiS2Filler(
        context_length=args.context_length,
        prediction_length=6,
        max_epochs=args.max_epochs,
        num_samples=args.num_samples,
        device=args.device,
        random_state=args.seed
    )
    
    # 3. Fill
    X_filled_m, diagnostics = filler.fit_predict_fill(
        X_visible_m=X_visible,
        target_quarter_end=target_q_end,
        cutoff_date=cutoff_date,
        dataset_type=args.dataset,
        target_quarter=args.target_quarter,
        vintage_label=args.vintage,
        exclude_cols=["gdp_target"],
        force_refit=args.force_refit
    )
    
    logger.info("Filling completed.")
    logger.info(f"Filled panel shape: {X_filled_m.shape}")
    
    # 4. Report details
    print("\n" + "="*40)
    print(" TACTiS-2 SMOKE TEST RESULTS")
    print("="*40)
    print(f"Number of origin groups: {diagnostics.get('tactis2_origin_groups')}")
    print(f"Origin dates used:      {diagnostics.get('tactis2_origin_dates_used')}")
    print(f"Values forecasted:      {diagnostics.get('tactis2_values_forecasted')}")
    print(f"Fallback ffill values:  {diagnostics.get('tactis2_values_ffill_fallback')}")
    print(f"Group failures:         {diagnostics.get('tactis2_group_failures')}")
    print("-" * 40)
    
    seed_suffix = f"_s{args.seed}"
    panel_name = f"X_filled_tactis2_{args.dataset}_{args.target_quarter}_{args.vintage}{seed_suffix}.parquet"
    audit_name = f"tactis2_filled_values_{args.dataset}_s{args.seed}.csv"
    
    print(f"Saved panel: data/forecasts/tactis2_filled_panels/{panel_name}")
    print(f"Saved audit: data/forecasts/tactis2_filled_values/{audit_name}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
