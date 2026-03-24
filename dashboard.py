# dashboard.py
import os
import logging
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("dashboard")

# ----------------------------
# DB
# ----------------------------
_DEFAULT_DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
DB_URL = os.environ.get("DB_URL", _DEFAULT_DB_URL)

engine = create_engine(
    DB_URL,
    future=True,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=5,
    connect_args={"connect_timeout": 10},
)

# ----------------------------
# Streamlit config
# ----------------------------
st.markdown("""
<style>
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .stMetric { padding: 15px; border-radius: 8px; border: 1px solid rgba(128,128,128,0.2); }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DB helpers (cached)
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def db_fetch_df(sql: str, params: Optional[dict] = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


@st.cache_data(ttl=300, show_spinner=False)
def fetch_providers() -> pd.DataFrame:
    return db_fetch_df("""
        SELECT id, name, base_url, created_at
        FROM providers
        ORDER BY name
    """)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_datasets(provider_id: Optional[int] = None, search: str = "", limit: int = 5000) -> pd.DataFrame:
    where = []
    params = {"limit": limit}

    if provider_id is not None:
        where.append("d.provider_id = :pid")
        params["pid"] = provider_id

    if search.strip():
        where.append("(d.key ILIKE :q OR d.title ILIKE :q OR d.description ILIKE :q)")
        params["q"] = f"%{search.strip()}%"

    wh = ("WHERE " + " AND ".join(where)) if where else ""
    return db_fetch_df(f"""
        SELECT d.id, d.provider_id, p.name AS provider, d.key, d.title, d.description, d.created_at
        FROM datasets d
        JOIN providers p ON p.id = d.provider_id
        {wh}
        ORDER BY p.name, d.key
        LIMIT :limit
    """, params)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_series(
    dataset_id: Optional[int] = None,
    search: str = "",
    limit: int = 300,
) -> pd.DataFrame:
    where = []
    params = {"limit": limit}

    if dataset_id is not None:
        where.append("s.dataset_id = :did")
        params["did"] = dataset_id

    if search.strip():
        where.append("(s.key ILIKE :q OR s.name ILIKE :q OR s.country ILIKE :q OR s.unit ILIKE :q)")
        params["q"] = f"%{search.strip()}%"

    wh = ("WHERE " + " AND ".join(where)) if where else ""
    return db_fetch_df(f"""
        SELECT
          s.id,
          s.dataset_id,
          s.key,
          s.name,
          s.country,
          s.frequency,
          s.transform,
          s.unit,
          s.created_at
        FROM series s
        {wh}
        ORDER BY s.id
        LIMIT :limit
    """, params)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_series_timespan(series_id: int) -> pd.DataFrame:
    return db_fetch_df("""
        SELECT
          MIN(o.period_date) AS min_date,
          MAX(o.period_date) AS max_date,
          COUNT(*)::bigint AS rows
        FROM observations o
        WHERE o.series_id = :sid
    """, {"sid": series_id})


@st.cache_data(ttl=300, show_spinner=False)
def fetch_observations(
    series_id: int,
    min_date: Optional[str] = None,
    max_rows: int = 200_000,
) -> pd.DataFrame:
    where = ["o.series_id = :sid"]
    params = {"sid": series_id, "limit": max_rows}

    if min_date:
        where.append("o.period_date >= :min_date")
        params["min_date"] = min_date

    wh = " AND ".join(where)

    df = db_fetch_df(f"""
        SELECT o.period_date AS date, o.value
        FROM observations o
        WHERE {wh}
        ORDER BY o.period_date
        LIMIT :limit
    """, params)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_health_overview() -> pd.DataFrame:
    # Optimized to group by provider only to prevent out-of-memory memory crashes on massive dataset counts
    return db_fetch_df("""
        SELECT
            p.name AS provider,
            COUNT(DISTINCT d.id)::bigint AS datasets_count,
            COUNT(DISTINCT s.id)::bigint AS series_count,
            MIN(o.period_date)  AS first_obs,
            MAX(o.period_date)  AS last_obs,
            MAX(il.run_time)    AS last_ingest
        FROM providers p
        LEFT JOIN datasets d ON d.provider_id = p.id
        LEFT JOIN series s ON s.dataset_id = d.id
        LEFT JOIN observations o ON o.series_id = s.id
        LEFT JOIN ingestion_log il ON il.dataset_id = d.id
        GROUP BY p.name
        ORDER BY p.name
    """)


def _fmt_dt(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "-"
    try:
        return pd.to_datetime(x).strftime("%Y-%m-%d")
    except Exception:
        return str(x)


# ============================================================
# UI blocks
# ============================================================
def render_catalog():
    st.title("📚 Data Catalog (universal)")
    st.caption("Shows everything that exists in DB: providers → datasets → series. Works automatically for new sources.")

    providers = fetch_providers()
    dataset_cnt = db_fetch_df("SELECT COUNT(*)::bigint AS n FROM datasets")["n"].iloc[0]
    series_cnt = db_fetch_df("SELECT COUNT(*)::bigint AS n FROM series")["n"].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Providers", int(len(providers)))
    c2.metric("Datasets", int(dataset_cnt))
    c3.metric("Series", int(series_cnt))

    st.subheader("Providers")
    st.dataframe(providers, use_container_width=True)

    st.subheader("Datasets (search)")
    q = st.text_input("Search datasets by key/title/description", value="", placeholder="e.g. gdp, hicp, LT, yahoo, ...")
    filtered = fetch_datasets(provider_id=None, search=q, limit=5000)
    st.write(f"Showing {len(filtered)} dataset(s) (limit 5000).")
    st.dataframe(filtered, use_container_width=True)


def render_dataset_explorer():
    st.title("🗂️ Dataset Explorer")
    st.caption("Pick a provider and a dataset, then search series inside that dataset.")

    providers = fetch_providers()
    if providers.empty:
        st.warning("No providers in DB.")
        return

    provider_name = st.selectbox("Provider", providers["name"].tolist(), index=0)
    provider_id = int(providers.loc[providers["name"] == provider_name, "id"].iloc[0])

    ds_search = st.text_input("Dataset search (within provider)", value="", placeholder="e.g. namq_10_gdp, GDPC1, ...")
    ds_df = fetch_datasets(provider_id=provider_id, search=ds_search, limit=2000)

    if ds_df.empty:
        st.info("No datasets match.")
        return

    # show a compact table
    st.write(f"Datasets found: {len(ds_df)} (limit 2000)")
    # select dataset by key
    ds_key = st.selectbox("Dataset key", ds_df["key"].tolist(), index=0)
    ds_row = ds_df.loc[ds_df["key"] == ds_key].iloc[0]
    dataset_id = int(ds_row["id"])

    st.markdown(
        f"""
**Selected dataset**
- **Provider**: `{ds_row['provider']}`
- **Key**: `{ds_row['key']}`
- **Title**: `{ds_row['title'] if pd.notna(ds_row['title']) else ''}`
"""
    )

    s_search = st.text_input("Series search (key/name/country/unit)", value="", placeholder="e.g. geo=LT, unit, unemployment ...")
    s_df = fetch_series(dataset_id=dataset_id, search=s_search, limit=400)

    if s_df.empty:
        st.info("No series found in this dataset (or your search is too strict).")
        return

    st.write(f"Series shown: {len(s_df)} (limit 400). If your dataset has thousands of series, refine search.")
    st.dataframe(s_df, use_container_width=True, height=300)

    # quick preview chart for one series
    st.subheader("Quick preview")
    series_id = st.selectbox("Pick series_id", s_df["id"].astype(int).tolist())
    s_row = s_df.loc[s_df["id"] == series_id].iloc[0]

    span = fetch_series_timespan(int(series_id))
    min_d = _fmt_dt(span["min_date"].iloc[0])
    max_d = _fmt_dt(span["max_date"].iloc[0])
    rows = int(span["rows"].iloc[0]) if pd.notna(span["rows"].iloc[0]) else 0

    st.markdown(
        f"""
**Series**
- **id**: `{int(s_row['id'])}`
- **key**: `{s_row['key']}`
- **name**: `{s_row['name'] if pd.notna(s_row['name']) else ''}`
- **country**: `{s_row['country']}` | **freq**: `{s_row['frequency']}` | **unit**: `{s_row['unit']}`
- **data span**: `{min_d}` → `{max_d}` | **rows**: `{rows:,}`
"""
    )

    min_date = st.text_input("Min date for chart (YYYY-MM-DD)", value="2004-01-01")
    max_rows = st.slider("Max rows to load (for this chart/table)", min_value=5_000, max_value=200_000, value=50_000, step=5_000)

    obs = fetch_observations(int(series_id), min_date=min_date.strip() or None, max_rows=int(max_rows))
    if obs.empty:
        st.warning("No observations returned (check min_date).")
        return

    fig = px.line(obs, x="date", y="value", title=str(s_row["name"] if pd.notna(s_row["name"]) else s_row["key"]))
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw observations table + download"):
        st.dataframe(obs, use_container_width=True, height=350)
        st.download_button(
            "Download CSV",
            data=obs.to_csv(index=False).encode("utf-8"),
            file_name=f"series_{series_id}.csv",
            mime="text/csv",
        )


def render_series_search():
    st.title("🔎 Series Search (universal)")
    st.caption("This avoids huge dropdowns. Search across ALL series by text and then open the chart.")

    q = st.text_input("Search series (key/name/country/unit)", value="", placeholder="e.g. GDP, LT, unemployment, HICP ...")
    limit = st.slider("Max search results", 50, 1000, 300, 50)

    if not q.strip():
        st.info("Type a query to search series.")
        return

    # Join to include provider/dataset quickly for context
    df = db_fetch_df("""
        SELECT
          s.id,
          p.name AS provider,
          d.key  AS dataset_key,
          s.key  AS series_key,
          s.name AS series_name,
          s.country,
          s.frequency,
          s.unit
        FROM series s
        JOIN datasets d ON d.id = s.dataset_id
        JOIN providers p ON p.id = d.provider_id
        WHERE (s.key ILIKE :q OR s.name ILIKE :q OR s.country ILIKE :q OR s.unit ILIKE :q OR d.key ILIKE :q OR p.name ILIKE :q)
        ORDER BY p.name, d.key, s.id
        LIMIT :limit
    """, {"q": f"%{q.strip()}%", "limit": int(limit)})

    st.write(f"Found {len(df)} result(s) (limit {limit}).")
    if df.empty:
        return

    st.dataframe(df, use_container_width=True, height=350)

    picked = st.selectbox("Pick a series_id to plot", df["id"].astype(int).tolist())
    row = df.loc[df["id"] == picked].iloc[0]

    span = fetch_series_timespan(int(picked))
    st.markdown(
        f"""
**Selected**
- **provider**: `{row['provider']}` | **dataset**: `{row['dataset_key']}`
- **series_id**: `{int(picked)}`
- **series_key**: `{row['series_key']}`
- **name**: `{row['series_name'] if pd.notna(row['series_name']) else ''}`
- **country**: `{row['country']}` | **freq**: `{row['frequency']}` | **unit**: `{row['unit']}`
- **span**: `{_fmt_dt(span['min_date'].iloc[0])}` → `{_fmt_dt(span['max_date'].iloc[0])}` | **rows**: `{int(span['rows'].iloc[0]) if pd.notna(span['rows'].iloc[0]) else 0:,}`
"""
    )

    min_date = st.text_input("Min date for chart (YYYY-MM-DD)", value="2004-01-01", key="series_search_min_date")
    max_rows = st.slider("Max rows to load", 5_000, 200_000, 50_000, 5_000, key="series_search_max_rows")

    obs = fetch_observations(int(picked), min_date=min_date.strip() or None, max_rows=int(max_rows))
    if obs.empty:
        st.warning("No observations returned.")
        return

    fig = px.line(obs, x="date", y="value", title=str(row["series_name"] if pd.notna(row["series_name"]) else row["series_key"]))
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def render_health():
    st.title("🩺 System Health")
    st.caption("Universal health view across all providers/datasets; shows time coverage and last ingests.")
    health = fetch_health_overview()
    if health.empty:
        st.info("No health data available.")
        return
    st.dataframe(health, use_container_width=True, height=550)


def render_nowcast():
    st.title("🧪 Experiment Runner")
    st.caption("Launch your existing model pipelines directly from the dashboard.")
    st.info("Pick a search or experiment script and run it. The results will automatically populate the Predictions tab.")

    scripts = [
        "run_tabular_search.py", "run_midas_search.py",
        "run_bridge_search.py", "run_dfm_experiment.py",
        "run_bvar_experiment.py", "run_tactis_search.py"
    ]
    
    script_to_run = st.selectbox("Select Model Pipeline:", scripts)
    fast_mode = st.checkbox("⚡ Fast / Demo Mode", value=True, help="Appends `--search-last-n-steps 3` and `--search-max-configs 2` to avoid hour-long gridsearches.")

    if st.button(f"🚀 Run {script_to_run}"):
        with st.spinner(f"Running {script_to_run}... (this could take minutes to hours depending on grid size)"):
            import subprocess
            try:
                cmd = ["python", f"scripts/{script_to_run}"]
                if fast_mode:
                    cmd.extend(["--search-last-n-steps", "3", "--search-max-configs", "2"])
                    
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                out = (result.stdout or "") + "\n" + (result.stderr or "")
                st.success(f"{script_to_run} finished successfully.")
                with st.expander("View Logs"):
                    st.code(out)
            except subprocess.CalledProcessError as e:
                out = (e.stdout or "") + "\n" + (e.stderr or "")
                st.error(f"{script_to_run} failed (non-zero exit code).")
                with st.expander("View Error Logs"):
                    st.code(out)
            except Exception as e:
                st.error(f"Execution failed: {e}")


def render_predictions():
    st.title("📈 Model Predictions & Leaderboard")
    st.markdown("Compare the best performing model configurations and track their real-time forecasts against actual data.", unsafe_allow_html=True)

    from pathlib import Path
    import glob
    base_dir = Path(".")

    # --- 1. LEADERBOARD ---
    st.header("🏆 Model Leaderboard")
    
    # Sweep entire project for result files, ignoring venvs
    grid_files = [f for f in base_dir.rglob("*gridsearch_*.csv") if "venv" not in str(f) and ".git" not in str(f)]
    
    leaderboard_data = []
    for fp in grid_files:
        try:
            df_g = pd.read_csv(fp)
            df_g_success = df_g[df_g["status"] == "success"]
            if not df_g_success.empty:
                best_idx = df_g_success["rmse"].idxmin()
                best_row = df_g_success.loc[best_idx]
                
                parts = fp.stem.split("_")
                mname = parts[1] if len(parts) > 1 else fp.stem
                model_name = str(best_row.get("model", mname)).upper()
                
                rmse = best_row.get("rmse", float('nan'))
                mae = best_row.get("mae", float('nan'))
                runtime = best_row.get("runtime_sec", float('nan'))
                
                leaderboard_data.append({
                    "Model Rank": model_name,
                    "RMSE": rmse,
                    "MAE": mae,
                    "Runtime (s)": f"{runtime:.1f}" if pd.notnull(runtime) else "N/A",
                    "Source": fp.name
                })
        except Exception:
            pass

    if leaderboard_data:
        ldb_df = pd.DataFrame(leaderboard_data).sort_values("RMSE").reset_index(drop=True)
        ldb_df.index = ldb_df.index + 1
        
        cols = st.columns(min(3, len(ldb_df)))
        for i, col in enumerate(cols):
            row = ldb_df.iloc[i]
            col.metric(label=f"🥇 #{i+1}: {row['Model Rank']}", value=f"RMSE: {row['RMSE']:.4f}", delta=f"MAE: {row['MAE']:.4f}", delta_color="off")
            
        st.dataframe(ldb_df.style.highlight_min(subset=["RMSE", "MAE"], color="#d4edda"), use_container_width=True)
    else:
        st.info("No gridsearch metrics found to build a leaderboard.")
        st.write("👉 Head over to the **Nowcast** tab to run your first experiment!")

    st.divider()

    # --- 2. FORECAST PLOT ---
    st.header("📊 Forecast Visualization")
    pred_files = [f for f in base_dir.rglob("*predictions_*.csv") if "venv" not in str(f) and ".git" not in str(f)]
    if not pred_files:
        st.info("No prediction CSVs found.")
        st.write("👉 Your previous gridsearch metric files were found, but older versions of your pipeline scripts did not save the actual time-series predictions. To see the interactive plot, please generate fresh predictions by running an experiment from the **Experiment Runner** tab.")
        return

    dfs = []
    for fp in pred_files:
        try:
            df_curr = pd.read_csv(fp)
            if "Target_Date" in df_curr.columns and "Predicted" in df_curr.columns:
                if "Model" not in df_curr.columns:
                    parts = fp.stem.split("_")
                    mname = parts[1] if len(parts) > 1 else fp.stem
                    df_curr["Model"] = mname.upper()
                dfs.append(df_curr)
        except Exception:
            pass
            
    if not dfs:
        st.warning("No valid prediction data found.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df["Target_Date"] = pd.to_datetime(combined_df["Target_Date"])
    combined_df = combined_df.sort_values("Target_Date")
    
    unique_models = sorted(combined_df["Model"].dropna().unique().tolist())
    
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("**Filter Models**")
        selected_models = []
        for m in unique_models:
            if st.checkbox(m, value=True):
                selected_models.append(m)
                
    with col2:
        if not selected_models:
            st.warning("Select at least one model to display the plot.")
        elif "Actual" in combined_df.columns and "Predicted" in combined_df.columns:
            plot_df = combined_df[combined_df["Model"].isin(selected_models)]
            
            df_actual = plot_df.drop_duplicates(subset=["Target_Date"])[["Target_Date", "Actual"]].sort_values("Target_Date")
            
            fig = px.line(df_actual, x="Target_Date", y="Actual", markers=True)
            if fig.data:
                fig.data[0].name = "Actual"
                # Use a theme-agnostic color for the Actual line, or let plotly decide
                fig.data[0].line.color = "#888888"
                fig.data[0].line.width = 3
            
            fig_pred = px.line(plot_df.sort_values("Target_Date"), x="Target_Date", y="Predicted", color="Model", markers=True)
            for trace in fig_pred.data:
                trace.line.dash = "dot"
                fig.add_trace(trace)
                
            fig.update_layout(
                hovermode="x unified",
                title="Actual vs Predicted Macroeconomic Trends",
                xaxis_title="Target Date",
                yaxis_title="Value",
                legend_title="Legend",
                # Removed plot_bgcolor="white" to support native dark mode
                xaxis={"showgrid": True, "rangeslider": {"visible": True}},
                yaxis={"showgrid": True}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            csv_data = plot_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Merged Predictions CSV",
                data=csv_data,
                file_name='merged_predictions.csv',
                mime='text/csv'
            )
        else:
            st.warning("Columns 'Actual' and 'Predicted' are required for the plot.")


# ============================================================
# Main
# ============================================================
def main():
    st.sidebar.title("Nowcast DB Dashboard")

    page = st.sidebar.radio(
        "Navigate",
        ["Catalog", "Dataset Explorer", "Series Search", "Health", "Nowcast", "Predictions"],
        index=5,
    )

    # connection hint
    with st.sidebar.expander("DB connection", expanded=False):
        st.code(DB_URL)

    if page == "Catalog":
        render_catalog()
    elif page == "Dataset Explorer":
        render_dataset_explorer()
    elif page == "Series Search":
        render_series_search()
    elif page == "Health":
        render_health()
    elif page == "Nowcast":
        render_nowcast()
    elif page == "Predictions":
        render_predictions()


if __name__ == "__main__":
    main()
