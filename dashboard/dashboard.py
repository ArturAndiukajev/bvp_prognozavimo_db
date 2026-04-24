# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from icalendar import Calendar
import os

# ============================================================
# 1. Page Configuration
# ============================================================
st.set_page_config(page_title="Lithuania GDP Nowcasting", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDataFrame th { color: #d62728 !important; font-weight: bold; }
    .intro-text { color: #555555; font-size: 14px; margin-bottom: 20px; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. Data Retrieval Functions
# ============================================================

@st.cache_data
def load_real_forecasts(model_name):
    """Reads forecasts (from CSV) and ACTUAL GDP (from Parquet)."""
    base_forecast_path = "/scratch/lustre/home/miva8802/nowcasting_project/database_code/data/forecasts/"
    base_data_path = "/scratch/lustre/home/miva8802/nowcasting_project/database_code/data/processed/"
    
    mapping = {
        "MIDAS (Mixed Frequency)": {
            "forecast": "FINAL_LONG_FORECAST_MIDAS.csv",
            "data": "mixed_final_nowcast_dataset.parquet"
        },
        "Elastic Net (Common Frequency)": {
            "forecast": "FINAL_LONG_FORECAST_ENET.csv",
            "data": "common_final_nowcast_dataset.parquet"
        }
    }
    
    selected_files = mapping.get(model_name, {})
    forecast_file = os.path.join(base_forecast_path, selected_files.get("forecast", ""))
    dataset_file = os.path.join(base_data_path, selected_files.get("data", ""))
    
    df_forecasts = pd.DataFrame()
    df_actual = pd.DataFrame()

    if os.path.exists(forecast_file):
        try:
            df = pd.read_csv(forecast_file)
            df = df.rename(columns={'Target_Date': 'forecast_date', 'Predicted': 'predicted_value'})
            df['forecast_date'] = pd.to_datetime(df['forecast_date'])
            df = df.dropna(subset=['predicted_value'])
            df = df.drop_duplicates(subset=['forecast_date'], keep='last')
            df_forecasts = df.sort_values('forecast_date')
        except Exception as e:
            st.error(f"Error reading forecasts: {e}")

    if os.path.exists(dataset_file):
        try:
            df_data = pd.read_parquet(dataset_file)
            if 'gdp_target' in df_data.columns:
                df_actual = df_data[['gdp_target']].dropna().reset_index()
                df_actual.columns = ['date', 'actual_gdp']
                df_actual['date'] = pd.to_datetime(df_actual['date'])
                df_actual = df_actual.sort_values('date')
        except Exception as e:
            st.error(f"Error reading original data: {e}")

    return df_forecasts, df_actual

@st.cache_data
def load_impact_results(model_name):
    """Reads variable impact CSV files."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    mapping = {
        "MIDAS (Mixed Frequency)": "midas_impact_results.csv",
        "Elastic Net (Common Frequency)": "elasticnet_impact_results.csv"
    }
    
    file_name = mapping.get(model_name, "")
    file_path = os.path.join(current_dir, file_name)
    
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            df = df.head(15)
            df['Coefficient'] = df['Coefficient'].round(5)
            if 'Abs_Impact' in df.columns:
                df = df.drop(columns=['Abs_Impact'])
            return df
        except Exception as e:
            st.error(f"Error reading impact data: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_real_upcoming_releases():
    """Downloads data from both Eurostat calendars."""
    URLS = [
        "https://ec.europa.eu/eurostat/o/calendars/eventsIcal?theme=0&category=1",
        "https://ec.europa.eu/eurostat/o/calendars/eventsIcal?theme=0&category=2"
    ]
    headers = {'User-Agent': 'Mozilla/5.0'}
    upcoming_events = []
    today = datetime.today().date()

    for url in URLS:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200: continue
            
            cal = Calendar.from_ical(response.text)
            for component in cal.walk('vevent'):
                dt = component.get('dtstart').dt
                release_date = dt.date() if hasattr(dt, 'date') else dt
                
                if release_date >= today:
                    summary = str(component.get('summary')).replace(" - EU and euro area", "").strip()
                    theme = str(component.get('X-THEME', 'Other')).capitalize()
                    
                    upcoming_events.append({
                        'Date': release_date, 
                        'Indicator': summary, 
                        'Theme': theme
                    })
        except: continue

    if not upcoming_events: return pd.DataFrame()

    df = pd.DataFrame(upcoming_events).drop_duplicates()
    df = df.sort_values('Date').head(25)
    df['Date'] = df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    return df

def style_coefficients(val):
    color = '#2ca02c' if val > 0 else '#d62728'
    return f'color: {color}; font-weight: bold;'


# ============================================================
# 3. UI Layout
# ============================================================
def main():
    # Sidebar removed as selections were placeholders

    st.markdown("<h2 style='color: #d62728;'>Lithuania Economic Nowcasting</h2>", unsafe_allow_html=True)
    st.markdown("<div class='intro-text'>This tool compares statistical Nowcasting methods to monitor Lithuania's GDP growth. Use the selection below to switch between forecasting models.</div>", unsafe_allow_html=True)
    
    # --- MODEL SELECTION ON MAIN PAGE ---
    model_choice = st.selectbox(
        "Select Forecasting Model:",
        ["MIDAS (Mixed Frequency)", "Elastic Net (Common Frequency)"],
        index=0
    )
    st.markdown("---")

    # --- Chart Section ---
    st.markdown(f"### Lithuania GDP Nowcasting")
    st.caption("The target variable represents the stationarized GDP quarterly growth rate after log-differencing and outlier winsorization.")
    
    df_forecasts, df_actual = load_real_forecasts(model_choice)

    if not df_forecasts.empty or not df_actual.empty:
        fig = go.Figure()

        if not df_actual.empty:
            fig.add_trace(go.Scatter(
                x=df_actual['date'],
                y=df_actual['actual_gdp'],
                mode='lines',
                name="Actual GDP",
                line=dict(color='grey', width=3)
            ))

        if not df_forecasts.empty:
            fig.add_trace(go.Scatter(
                x=df_forecasts['forecast_date'],
                y=df_forecasts['predicted_value'],
                mode='lines', 
                name="Model Forecast",
                line=dict(color='#d62728', width=2.5) 
            ))

        fig.update_layout(
            hovermode="x unified",
            plot_bgcolor="white",
            xaxis_title="",
            yaxis_title="Growth Rate",
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Failed to load forecast or actual data. Please check file paths.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Bottom Tables ---
    col1, col2 = st.columns(2)
    
    # Left Side: Model Feature Impact
    with col1:
        st.markdown(f"<h4 style='color: #d62728;'>Feature Impact ({model_choice.split()[0]})</h4>", unsafe_allow_html=True)
        st.caption("Relative importance of each macroeconomic indicator in determining the current GDP nowcast.")
        
        df_impact = load_impact_results(model_choice)
        
        if not df_impact.empty:
            display_df = df_impact[['Name', 'Coefficient']].copy()
            display_df.columns = ['Indicator', 'Weight']
            
            styled_df = display_df.style.map(style_coefficients, subset=['Weight'])
            
            event = st.dataframe(
                styled_df,
                width="stretch",
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun"
            )
            
            if len(event.selection.rows) > 0:
                selected_index = event.selection.rows[0]
                selected_row = df_impact.iloc[selected_index]
                
                st.markdown("---")
                st.subheader("Indicator Details")
                st.markdown(f"**Name:** {selected_row['Name']}")
                st.markdown(f"**Eurostat Code:** `{selected_row.get('Series_Key', 'N/A')}`")
                
                if 'Variable' in selected_row:
                    st.markdown(f"**System ID:** `{selected_row['Variable']}`")
                
                weight_color = "#2ca02c" if selected_row['Coefficient'] > 0 else "#d62728"
                st.markdown(f"**Impact Weight:** <span style='color:{weight_color}; font-weight:bold;'>{selected_row['Coefficient']}</span>", unsafe_allow_html=True)
                
            else:
                st.info("Select an indicator in the table to see details.")
        else:
            st.info("No impact data found.")
            
    # Right Side: Eurostat Calendar
    with col2:
        st.markdown("<h4 style='color: #d62728;'>Upcoming Releases</h4>", unsafe_allow_html=True)
        st.caption("Data and Euro indicator releases from Eurostat release calndar.")

        df_upcoming = fetch_real_upcoming_releases()
        if not df_upcoming.empty:
            st.dataframe(
                df_upcoming, 
                width="stretch",
                hide_index=True
            )
            

if __name__ == "__main__":
    main()