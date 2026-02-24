import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import text
from scripts import db_helpers
from datetime import datetime, timezone

# Page config
st.set_page_config(page_title="Nowcast Data Dashboard", layout="wide")

@st.cache_resource
def get_db_engine():
    return db_helpers.get_engine()

def get_providers(engine):
    with engine.connect() as conn:
        res = conn.execute(text("SELECT id, name FROM providers ORDER BY name"))
        return pd.DataFrame(res.fetchall(), columns=["id", "name"])

def get_datasets(engine, provider_id):
    with engine.connect() as conn:
        res = conn.execute(text("SELECT id, key, title FROM datasets WHERE provider_id = :pid ORDER BY key"), {"pid": provider_id})
        return pd.DataFrame(res.fetchall(), columns=["id", "key", "title"])

def get_series(engine, dataset_id):
    with engine.connect() as conn:
        res = conn.execute(text("""
            SELECT id, key, country, frequency, transform, unit, name 
            FROM series 
            WHERE dataset_id = :did 
            ORDER BY key
        """), {"did": dataset_id})
        return pd.DataFrame(res.fetchall(), columns=["id", "key", "country", "frequency", "transform", "unit", "name"])

def get_observations(engine, series_id):
    with engine.connect() as conn:
        res = conn.execute(text("""
            SELECT period_date, value, observed_at
            FROM observations
            WHERE series_id = :sid
            ORDER BY period_date ASC, observed_at DESC
        """), {"sid": series_id})
        df = pd.DataFrame(res.fetchall(), columns=["date", "value", "observed_at"])
        if not df.empty:
            # For visualization, we usually want the latest vintage for each period
            df = df.sort_values(["date", "observed_at"], ascending=[True, False]).drop_duplicates("date")
        return df

def main():
    st.title("📊 Nowcast Data Dashboard")
    
    engine = get_db_engine()
    
    # Sidebar
    st.sidebar.header("Data Selection")
    
    providers_df = get_providers(engine)
    if providers_df.empty:
        st.warning("No providers found in the database. Run ingestion first.")
        return

    selected_provider_name = st.sidebar.selectbox("Select Provider", providers_df["name"])
    provider_id = providers_df[providers_df["name"] == selected_provider_name]["id"].values[0]
    
    datasets_df = get_datasets(engine, int(provider_id))
    if datasets_df.empty:
        st.info("No datasets found for this provider.")
        return

    # Create a nice label for datasets
    datasets_df["label"] = datasets_df.apply(lambda r: f"{r['key']} - {r['title']}" if r['title'] else r['key'], axis=1)
    selected_dataset_label = st.sidebar.selectbox("Select Dataset", datasets_df["label"])
    dataset_id = datasets_df[datasets_df["label"] == selected_dataset_label]["id"].values[0]
    
    series_df = get_series(engine, int(dataset_id))
    if series_df.empty:
        st.info("No series found for this dataset.")
        return

    # Series Selector
    series_df["label"] = series_df.apply(lambda r: f"{r['key']} ({r['country']}, {r['frequency']})", axis=1)
    selected_series_label = st.sidebar.selectbox("Select Series", series_df["label"])
    series_row = series_df[series_df["label"] == selected_series_label].iloc[0]
    series_id = series_row["id"]
    
    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualization", "📋 Data Table", "🏥 Health Check", "🔮 Nowcast"])
    
    with tab1:
        st.subheader(f"Series: {series_row['name'] or series_row['key']}")
        
        with st.spinner("Fetching observations..."):
            obs_df = get_observations(engine, int(series_id))
        
        if obs_df.empty:
            st.info("No observations found for this series.")
        else:
            # Controls for transform
            transform_opt = st.radio("Display Mode", ["Levels", "YoY Growth (%)", "MoM/QoQ Growth (%)"], horizontal=True)
            
            plot_df = obs_df.copy()
            plot_df["date"] = pd.to_datetime(plot_df["date"])
            plot_df = plot_df.sort_values("date")
            
            y_col = "value"
            title = f"{selected_series_label} - Levels"
            
            if transform_opt == "YoY Growth (%)":
                # Assuming monthly or quarterly
                periods = 12 if series_row["frequency"] == "M" else 4
                plot_df["growth"] = plot_df["value"].pct_change(periods=periods) * 100
                y_col = "growth"
                title = f"{selected_series_label} - Year-over-Year Growth (%)"
            elif transform_opt == "MoM/QoQ Growth (%)":
                plot_df["growth"] = plot_df["value"].pct_change() * 100
                y_col = "growth"
                label = "MoM" if series_row["frequency"] == "M" else "QoQ"
                title = f"{selected_series_label} - {label} Growth (%)"
            
            fig = px.line(plot_df, x="date", y=y_col, title=title, markers=True)
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **Metadata**:
            - **Key**: `{series_row['key']}`
            - **Unit**: `{series_row['unit']}`
            - **Frequency**: `{series_row['frequency']}`
            - **Last Obs Date**: `{obs_df['date'].max()}`
            """)

    with tab2:
        st.subheader("Raw Observations")
        if 'obs_df' in locals() and not obs_df.empty:
            st.dataframe(obs_df, use_container_width=True)
            csv = obs_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"{series_row['key']}_data.csv",
                mime='text/csv',
            )
        else:
            st.info("No data available.")

    with tab3:
        st.subheader("System Health Overview")
        with engine.connect() as conn:
            res = conn.execute(text("""
                SELECT 
                    p.name AS provider,
                    d.key AS dataset,
                    count(s.id) AS series_count,
                    max(il.run_time) AS last_run
                FROM providers p
                JOIN datasets d ON d.provider_id = p.id
                LEFT JOIN series s ON s.dataset_id = d.id
                LEFT JOIN ingestion_log il ON il.dataset_id = d.id
                GROUP BY p.name, d.key
                ORDER BY p.name, d.key
            """))
            health_df = pd.DataFrame(res.fetchall(), columns=["Provider", "Dataset", "Series Count", "Last Ingested"])
            st.table(health_df)
            
    with tab4:
        st.subheader("🔮 Real-Time GDP Nowcast")
        st.write("Current model: Dynamic Factor Model (PCA-based) for US Real GDP.")
        
        if st.button("Run Nowcast Engine"):
            with st.spinner("Running nowcast pipeline..."):
                try:
                    from scripts.run_nowcast import main as execute_nowcast
                    # For demo, we'll just run a slightly modified version or shell out
                    # Here we show the latest result based on the integrated script
                    import subprocess
                    result = subprocess.run(["py", "-m", "scripts.run_nowcast"], capture_output=True, text=True, check=True)
                    
                    # Extract result from output
                    out = result.stderr + result.stdout
                    if "NOWCAST" in out:
                        val = out.split("NOWCAST (GDP Growth):")[-1].split("%")[0].strip()
                        st.success(f"### Latest Nowcast: {val}% GDP Growth")
                    else:
                        st.error("Nowcast completed but result not found in output.")
                        st.code(out)
                except Exception as e:
                    st.error(f"Failed to run nowcast: {e}")
        
        st.info("""
        **Methodology**:
        - Automatic stationarity transforms (Log-Diff) for 100+ indicators.
        - Common factor extraction via Principal Component Analysis (PCA).
        - OLS bridging to Quarterly Real GDP.
        - Handles 'ragged edges' via mean imputation of the most recent month data.
        """)

if __name__ == "__main__":
    main()
