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
st.set_page_config(page_title="Nowcast Dashboard", layout="wide")


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
    return db_fetch_df("""
        SELECT
            p.name AS provider,
            d.key  AS dataset,
            COUNT(DISTINCT s.id)::bigint AS series_count,
            MIN(o.period_date) AS first_obs,
            MAX(o.period_date) AS last_obs,
            MAX(il.run_time)   AS last_ingest
        FROM providers p
        JOIN datasets d ON d.provider_id = p.id
        LEFT JOIN series s ON s.dataset_id = d.id
        LEFT JOIN observations o ON o.series_id = s.id
        LEFT JOIN ingestion_log il ON il.dataset_id = d.id
        GROUP BY p.name, d.key
        ORDER BY p.name, d.key
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
    datasets = fetch_datasets(provider_id=None, search="", limit=100000)

    c1, c2, c3 = st.columns(3)
    c1.metric("Providers", int(len(providers)))
    c2.metric("Datasets", int(len(datasets)))
    # series count can be huge; query cheaply:
    series_cnt = db_fetch_df("SELECT COUNT(*)::bigint AS n FROM series")["n"].iloc[0]
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
    st.title("🔮 Nowcast Runner")
    st.caption("Runs your existing nowcast script and displays its output.")
    st.info("This tab does NOT implement the model. It just launches your existing pipeline script.")

    st.write("Script: `scripts/run_nowcast.py` (called via `py -m scripts.run_nowcast`)")

    if st.button("Run nowcast pipeline"):
        with st.spinner("Running nowcast..."):
            try:
                import subprocess
                result = subprocess.run(
                    ["py", "-m", "scripts.run_nowcast"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                out = (result.stdout or "") + "\n" + (result.stderr or "")
                st.success("Nowcast finished.")
                st.code(out)
            except subprocess.CalledProcessError as e:
                out = (e.stdout or "") + "\n" + (e.stderr or "")
                st.error("Nowcast failed (non-zero exit code).")
                st.code(out)
            except Exception as e:
                st.error(f"Nowcast failed: {e}")


# ============================================================
# Main
# ============================================================
def main():
    st.sidebar.title("Nowcast DB Dashboard")

    page = st.sidebar.radio(
        "Navigate",
        ["Catalog", "Dataset Explorer", "Series Search", "Health", "Nowcast"],
        index=1,
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


if __name__ == "__main__":
    main()
