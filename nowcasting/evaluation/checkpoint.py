import time
import json
import hashlib
from pathlib import Path
import pandas as pd
import logging
from joblib import Parallel, delayed

logger = logging.getLogger("checkpoint")

def build_run_signature(**kwargs) -> dict:
    return {k: v for k, v in kwargs.items() if v is not None}

def manifest_path_for(out_path: Path) -> Path:
    return out_path.with_suffix('.manifest.json')

def save_run_manifest(out_path: Path, signature: dict) -> None:
    manifest_path = manifest_path_for(out_path)
    with open(manifest_path, 'w') as f:
        json.dump(signature, f, indent=4, sort_keys=True)

def load_run_manifest(out_path: Path) -> dict | None:
    manifest_path = manifest_path_for(out_path)
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            return json.load(f)
    return None

def validate_resume_signature(out_path: Path, signature: dict, resume: bool) -> None:
    if not resume:
        save_run_manifest(out_path, signature)
        return
        
    loaded_sig = load_run_manifest(out_path)
    if loaded_sig is None:
        raise ValueError(f"Resume refused: Missing manifest for existing results at {out_path}. Cannot guarantee compatibility.")
        
    mismatches = []
    for k, v in signature.items():
        if k not in loaded_sig or loaded_sig[k] != v:
            mismatches.append(f"{k} (new: {v}, old: {loaded_sig.get(k)})")
            
    if mismatches:
        raise ValueError(f"Resume refused: Existing checkpoint was created with a different context. Mismatches: {', '.join(mismatches)}")

def get_config_hash(cfg: dict, active_steps: int, run_signature: dict | None = None) -> str:
    cfg_copy = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    cfg_copy["_search_last_n_steps"] = active_steps
    if run_signature is not None:
        cfg_copy["_run_signature"] = run_signature
    cfg_str = json.dumps(cfg_copy, sort_keys=True)
    return hashlib.md5(cfg_str.encode()).hexdigest()

def prune_grid(current_grid, active_steps, out_path, resume=False, run_signature=None):
    if run_signature is not None:
        if resume and not out_path.exists():
            raise ValueError(
                f"Resume refused: results file does not exist at {out_path}. "
                "Cannot resume a run that has no checkpoint CSV."
            )
        validate_resume_signature(out_path, run_signature, resume)

    if not (resume and out_path.exists()):
        for c in current_grid:
            c["_config_id"] = get_config_hash(c, active_steps, run_signature)
        return current_grid
        
    try:
        df_exist = pd.read_csv(out_path)
        if "status" in df_exist.columns and "_config_id" in df_exist.columns:
            completed = set(df_exist.loc[df_exist["status"] == "success", "_config_id"])
        else:
            completed = set()
    except Exception as e:
        logger.warning(f"Could not read existing results for resume: {e}")
        completed = set()

    pruned = []
    for c in current_grid:
        c["_config_id"] = get_config_hash(c, active_steps, run_signature)
        if c["_config_id"] not in completed:
            pruned.append(c)
            
    logger.info(f"Resume: skipped {len(current_grid) - len(pruned)} completed configs. {len(pruned)} remaining.")
    return pruned


def run_chunks(search_grid, run_fn, run_kwargs, out_path, n_jobs=1, chunk_sz=10, stage_name=""):
    res_all = []
    
    for i in range(0, len(search_grid), chunk_sz):
        chunk = search_grid[i:i+chunk_sz]
        if n_jobs > 1:
            res_chunk = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(run_fn)(cfg, **run_kwargs) for cfg in chunk
            )
        else:
            res_chunk = []
            for j, cfg in enumerate(chunk, 1):
                logger.info(f"[{i+j}/{len(search_grid)}] {stage_name} Running: {cfg}")
                res_chunk.append(run_fn(cfg, **run_kwargs))
        
        res_all.extend(res_chunk)
        
        # Immediately append to CSV
        df_chunk = pd.DataFrame(res_chunk)
        if 'selector_params' in df_chunk.columns:
            df_chunk['selector_params'] = df_chunk['selector_params'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
        if 'regression_params' in df_chunk.columns:
            df_chunk['regression_params'] = df_chunk['regression_params'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
            
        df_out_chunk = df_chunk.drop(columns=["_eval_df"]) if "_eval_df" in df_chunk.columns else df_chunk
        
        mode = "a" if out_path.exists() else "w"
        write_header = not out_path.exists()
        df_out_chunk.to_csv(out_path, mode=mode, header=write_header, index=False)
        
    return res_all
