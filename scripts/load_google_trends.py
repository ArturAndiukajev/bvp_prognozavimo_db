import logging
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import create_engine, text
from pytrends.request import TrendReq

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DB
DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
engine = create_engine(DB_URL, future=True)

# Config
TIMEFRAME = "2004-01-01 2025-03-31"
COUNTRIES = {
    "LT": "LT",
    "US": "US"
}

KEYWORDS = [
    "crisis", "Auto insurance", "house prices", "Immigration", "inflation", "business",
    "tourism", "credit", "prices", "export", "import", "Energy", "work", "Loan",
    "Food prices", "unemployment", "Economy", "housing crisis", "emigration",
    "fuel price", "Circular economy", "recession", "slowdown", "recovery",
    "manufacturing", "production", "supply chain", "demand", "cost of living",
    "consumer sentiment", "consumer confidence", "disposable income", "wage growth",
    "wage stagnation", "wage cuts", "layoffs", "bankruptcy", "insolvency",
    "housing market", "real estate", "mortgage rates", "interest rate",
    "central bank", "monetary policy", "fiscal policy", "government debt",
    "budget deficit", "inflation expectations", "commodity prices", "raw materials",
    "industrial output", "retail sales", "e-commerce growth", "digital transformation",
    "investment", "foreign direct investment", "capital flows", "exchange rate",
    "currency depreciation", "trade war", "tariffs", "sanctions",
    "supply disruption", "logistics bottleneck", "stock market crash",
    "asset bubble", "over-capacity", "under-employment", "shadow economy",
    "gray economy", "corporate profits", "margin squeeze", "deflation",
    "stagflation", "credit crunch", "bank run", "non-performing loans",
    "financial stability", "wealth inequality", "social unrest", "migration flows",
    "demographic change", "ageing population", "labor shortage", "automation",
    "green transition", "climate risk", "energy transition", "resource depletion",
    "housing affordability", "rental market", "housing bubble", "transportation cost",
    "logistics cost", "infrastructure investment", "public investment",
    "private consumption", "consumer spending", "business confidence",
    "corporate investment"
]


def ensure_source(conn):
    conn.execute(text("""
        insert into sources (name)
        values ('google_trends')
        on conflict (name) do nothing
    """))
    return conn.execute(
        text("select id from sources where name='google_trends'")
    ).scalar_one()


def ensure_series(conn, source_id, keyword, country):
    key = f"google_trends.{keyword}"

    return conn.execute(text("""
        insert into series (source_id, key, country, frequency, transform, unit, name, meta)
        values (:sid, :key, :country, 'M', 'LEVEL', 'INDEX_0_100', :name, '{}'::jsonb)
        on conflict (source_id, key, country, frequency, transform)
        do update set name = excluded.name
        returning id
    """), {
        "sid": source_id,
        "key": key,
        "country": country,
        "name": f"Google Trends: {keyword}"
    }).scalar_one()


def insert_observations(conn, series_id, df):
    observed_at = datetime.now(timezone.utc)

    for date, row in df.iterrows():
        value = row["value"]

        if pd.isna(value):
            continue

        conn.execute(text("""
            insert into observations (series_id, period_date, observed_at, value, meta)
            values (:sid, :date, :obs, :val, '{}'::jsonb)
            on conflict (series_id, period_date, observed_at) do nothing
        """), {
            "sid": series_id,
            "date": date.date(),
            "obs": observed_at,
            "val": float(value)
        })


def monthly_from_weekly(df):
    df.index = pd.to_datetime(df.index)
    monthly = df.resample("MS").mean()
    monthly = monthly.rename(columns={monthly.columns[0]: "value"})
    return monthly


def chunk_keywords(lst, size=5):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def fetch_trends():
    pytrends = TrendReq()

    with engine.begin() as conn:
        source_id = ensure_source(conn)

        for country_name, geo in COUNTRIES.items():
            logger.info(f"Country: {country_name}")

            for batch in chunk_keywords(KEYWORDS, 5):
                logger.info(f"Batch: {batch}")

                try:
                    pytrends.build_payload(batch, timeframe=TIMEFRAME, geo=geo)
                    data = pytrends.interest_over_time()

                    if data.empty:
                        continue

                    data = data.drop(columns=["isPartial"], errors="ignore")

                    for keyword in batch:
                        if keyword not in data.columns:
                            continue

                        df_kw = data[[keyword]].rename(columns={keyword: "value"})
                        df_kw = monthly_from_weekly(df_kw)

                        series_id = ensure_series(conn, source_id, keyword, country_name)
                        insert_observations(conn, series_id, df_kw)

                except Exception as e:
                    logger.warning(f"Failed batch {batch}: {e}")


if __name__ == "__main__":
    fetch_trends()