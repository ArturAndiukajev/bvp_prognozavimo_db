# GDP Nowcasting System

A macroeconomic nowcasting system built with Python and PostgreSQL. This tool is designed to handle mixed-frequency and common-frequency data, perform backtesting without look-ahead bias, and compare various forecasting methodologies (DFM, BVAR, MIDAS, ML, Deep Learning) using automated hyperparameter optimization.

## Key Features
- **ETL Pipeline**: Automated data ingestion from Eurostat, FRED (ALFRED), StatGov, Yahoo Finance, and Google Trends.
- **Mixed Frequency Support**: Dedicated models and data preparation for combining monthly and quarterly data.
- **Robust Backtesting**: Forward-rolling evaluation framework ensuring no data leakage.
- **Dynamic Factor Models (DFM)**: latent factor extraction for high-dimensional macroeconomic datasets.
- **Deep Learning**: Integration with TACTiS (Transformer-based architecture).
- **Dashboard**: Interactive Streamlit interface for results visualization and model monitoring.

---

## Quick Start Guide

### 1. Environment Setup
Create a PostgreSQL database and configure your `.env` file. You will need a FRED API key for ALFRED data.

**How to get ALFRED API Key:**
1. Create an account at [St. Louis Fed](https://fredaccount.stlouisfed.org/apikeys).
2. Generate your API key.
3. Add it to your `.env` file:
   ```env
   DB_URL=postgresql+psycopg2://user:password@localhost:5432/nowcast_db
   FRED_API_KEY=your_key_here
   ```

### 2. Initialize Database
Initialize the schema and tables.
```bash
python create_schema.py
```
*Note: This will wipe existing data in the database.*

### 3. Data Ingestion
Perform the initial data download.
```bash
python run_ingest.py --mode initial
```
For regular daily/weekly updates:
```bash
python run_updates.py
```

### 4. Data Preparation
Transform raw database records into processed Parquet files suitable for modeling.
```bash
python scripts/data_preparation.py --mode common_frequency --target GDP1
```

### 5. Training & Backtesting
Run experiments using the provided runners.
```bash
python experiments/run_dfm_experiment.py --engine optuna
```
Use `--resume` to continue interrupted experiments.

### 6. Visualization
Launch the dashboard to analyze forecasts and metrics.
```bash
streamlit run dashboard.py
```

---

## Configuration

- **`config/datasets.yaml`**: The central registry for data sources. Add or remove series IDs from Eurostat, FRED, etc., here.
- **`alembic.ini`**: Configuration for database migrations.

## Principal Control Scripts

- **`docker-compose.yml`**: Spins up a local PostgreSQL instance.
- **`create_schema.py`**: Initializes the DB schema.
- **`scripts/clear_data.py`**: Performs a full database wipe using CASCADE login.
- **`full_reload.py`**: Orchestrates a full system initialization (schema + initial ingest).
- **`run_updates.py`**: Incremental update mode (hashes API responses to avoid duplicates).
- **`run_ingest.py`**: Parallelized data ingestion runner.

## Data Loaders (scripts/)

- **`load_alfred.py`**: Real-time vintages from ALFRED. Handles revisions.
- **`load_fredmd.py`**: Snapshot-based FRED-MD panel ingestion.
- **`load_eurostat.py`**: Bulk download of Eurostat datasets based on filters.
- **`load_google_trends.py`**: Ingests Google Trends data with snapshotting.
- **`load_financials.py`**: Tail-based updates from Yahoo Finance using `yfinance`.

---

## Core Architecture

### Data Preparation (`scripts/data_preparation.py`)
Automates cleaning, YoY transformation, and stationarity testing (KPSS). It supports:
- **Common Frequency Mode**: Aggregates all data to a single target frequency (e.g., Monthly).
- **Mixed Frequency Mode**: Preserves original frequencies for models like MIDAS.
- **Ragged Edge Handling**: Smart imputation of publication lags at the end of series.

### Backtesting (`nowcasting/evaluation/backtester.py`)
A strict out-of-sample forward-rolling evaluator. It ensures that only information available at each historical point is used for prediction.

### Models
Supported architectures include:
- **Dynamic Factor Models (DFM)**: Standard and Mixed-Frequency implementations via Kalman Filter.
- **Bayesian VAR (BVAR)**: Frequentist and Bayesian VAR with Minnesota priors.
- **MIDAS**: Directly regressing low-frequency targets on high-frequency lags.
- **Bridge Equations**: Two-step mixed-frequency integration.
- **ML Regressions**: LightGBM and ElasticNet based models.
- **TACTiS**: Transformer-based multivariate time series model.

---

## Quality & Monitoring
- **`check_health.py`**: Monitors data staleness, gaps, and data corruption (flatlines).
- **`scripts/get_vintage.py`**: Utility to extract data state as it existed on a specific historical date.
- **`check_db.py`**: Diagnostic script to count records and verify ingestion success.

