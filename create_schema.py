import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
DB_URL = os.environ.get("DB_URL", _DEFAULT_DB_URL)
engine = create_engine(
    DB_URL,
    future=True,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 10},
)

DDL = """
-- Drop all tables in dependency order
DROP TABLE IF EXISTS ingestion_log CASCADE;
DROP TABLE IF EXISTS observations CASCADE;
DROP TABLE IF EXISTS releases CASCADE;
DROP TABLE IF EXISTS series CASCADE;
DROP TABLE IF EXISTS datasets CASCADE;
DROP TABLE IF EXISTS providers CASCADE;

-- ============================================================
-- providers = original source system
--   e.g. 'eurostat', 'alfred', 'yahoo_finance', 'google_trends', 'fredmd'
-- ============================================================
CREATE TABLE providers (
  id          BIGSERIAL PRIMARY KEY,
  name        TEXT NOT NULL UNIQUE,
  base_url    TEXT,
  meta        JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- datasets = one logical dataset inside a provider
--   eurostat: table code  (e.g. 'nama_10_gdp')
--   alfred:   FRED series (e.g. 'GDP', 'CPI')
--   yahoo:    ticker      (e.g. '^GSPC')
--   google:   geo name    (e.g. 'Lithuania', 'United States')
--   fredmd:   'fredmd_monthly_current'
-- ============================================================
CREATE TABLE datasets (
  id            BIGSERIAL PRIMARY KEY,
  provider_id   BIGINT NOT NULL REFERENCES providers(id) ON DELETE CASCADE,

  key           TEXT NOT NULL,        -- stable identifier within provider
  title         TEXT,                 -- human-readable title (optional)
  description   TEXT,
  meta          JSONB NOT NULL DEFAULT '{}'::jsonb,

  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE (provider_id, key)
);

-- ============================================================
-- series = one observable time series inside a dataset
-- ============================================================
CREATE TABLE series (
  id          BIGSERIAL PRIMARY KEY,
  dataset_id  BIGINT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,

  key         TEXT NOT NULL,           -- series identifier within dataset
  country     TEXT NOT NULL,
  frequency   TEXT NOT NULL,           -- 'D','W','M','Q','A'
  transform   TEXT NOT NULL,           -- 'LEVEL','LOG','DIFF', …
  unit        TEXT,
  name        TEXT,
  meta        JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE (dataset_id, key, country, frequency, transform)
);

-- ============================================================
-- releases = an ingestion event (download snapshot / vintage)
-- ============================================================
CREATE TABLE releases (
  id            BIGSERIAL PRIMARY KEY,
  dataset_id    BIGINT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,

  release_time  TIMESTAMPTZ NOT NULL,
  downloaded_at TIMESTAMPTZ NOT NULL,
  vintage_at    TIMESTAMPTZ NOT NULL,

  description   TEXT,
  raw_path      TEXT,
  content_hash  TEXT,
  meta          JSONB NOT NULL DEFAULT '{}'::jsonb,

  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- observations = individual data points
-- ============================================================
CREATE TABLE observations (
  id          BIGSERIAL PRIMARY KEY,
  series_id   BIGINT NOT NULL REFERENCES series(id) ON DELETE CASCADE,
  period_date DATE NOT NULL,
  observed_at TIMESTAMPTZ NOT NULL,
  value       DOUBLE PRECISION NOT NULL,

  -- MINIMUM REQUIRED CHANGE (for loader compatibility)
  status      TEXT,

  release_id  BIGINT REFERENCES releases(id) ON DELETE SET NULL,
  meta        JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE (series_id, period_date, observed_at)
);

-- ============================================================
-- ingestion_log = per-dataset run audit trail
-- ============================================================
CREATE TABLE ingestion_log (
  id            BIGSERIAL PRIMARY KEY,
  dataset_id    BIGINT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  run_time      TIMESTAMPTZ NOT NULL DEFAULT now(),
  status        TEXT NOT NULL,
  rows_inserted INTEGER NOT NULL DEFAULT 0,
  rows_failed   INTEGER NOT NULL DEFAULT 0,
  details       JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- ============================================================
-- Indexes
-- ============================================================
CREATE INDEX ix_datasets_provider_id        ON datasets(provider_id);
CREATE INDEX ix_series_dataset_id           ON series(dataset_id);
CREATE INDEX ix_observations_series_period  ON observations(series_id, period_date);
CREATE INDEX ix_observations_series_vintage ON observations(series_id, observed_at);
CREATE INDEX ix_releases_dataset_vintage    ON releases(dataset_id, vintage_at);
CREATE INDEX ix_releases_dataset_downloaded ON releases(dataset_id, downloaded_at DESC);
CREATE INDEX ix_releases_meta_scope         ON releases ((meta->>'scope'));
CREATE INDEX ix_observations_release_id     ON observations(release_id);
CREATE INDEX ix_ingestion_log_dataset_time  ON ingestion_log(dataset_id, run_time DESC);
CREATE INDEX ix_observations_series_latest  ON observations(series_id, period_date DESC, observed_at DESC);
"""


def main():
    with engine.begin() as conn:
        conn.execute(text(DDL))
    print("Schema created successfully.")


if __name__ == "__main__":
    main()



# from sqlalchemy import create_engine, text

# DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
# engine = create_engine(DB_URL, future=True)

# DDL = """
# -- Clean drop (order matters with FKs, CASCADE simplifies)
# DROP TABLE IF EXISTS ingestion_log CASCADE;
# DROP TABLE IF EXISTS observations CASCADE;
# DROP TABLE IF EXISTS releases CASCADE;
# DROP TABLE IF EXISTS series CASCADE;
# DROP TABLE IF EXISTS datasets CASCADE;
# DROP TABLE IF EXISTS providers CASCADE;

# -- Providers = original source system (stat_gov, eurostat, alfred, yahoo_finance, google_trends, etc.)
# CREATE TABLE providers (
#   id          BIGSERIAL PRIMARY KEY,
#   name        TEXT NOT NULL UNIQUE,   -- e.g. 'stat_gov', 'eurostat', 'alfred', ...
#   base_url    TEXT,                  -- optional: API root / homepage
#   meta        JSONB NOT NULL DEFAULT '{}'::jsonb,
#   created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
# );

# -- Datasets = one "table"/"flow"/"dataset id" inside a provider
# -- This replaces your old 'sources' table.
# CREATE TABLE datasets (
#   id            BIGSERIAL PRIMARY KEY,
#   provider_id   BIGINT NOT NULL REFERENCES providers(id) ON DELETE CASCADE,

#   -- stable dataset identifier within provider:
#   -- examples:
#   --   stat_gov: flow_id like 'S3R629_M3010217'
#   --   eurostat: table like 'nama_10_gdp'
#   --   fredmd:   vintage or file version
#   key           TEXT NOT NULL,

#   title         TEXT,                -- optional human title (EN preferred if you want)
#   description   TEXT,
#   meta          JSONB NOT NULL DEFAULT '{}'::jsonb,

#   created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

#   UNIQUE (provider_id, key)
# );

# CREATE TABLE series (
#   id          BIGSERIAL PRIMARY KEY,
#   dataset_id  BIGINT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,

#   key         TEXT NOT NULL,          -- series identifier WITHIN dataset (or full series key)
#   country     TEXT NOT NULL,
#   frequency   TEXT NOT NULL,          -- 'D','W','M','Q','A'
#   transform   TEXT NOT NULL,          -- 'LEVEL','LOG','DIFF',...
#   unit        TEXT,
#   name        TEXT,
#   meta        JSONB NOT NULL DEFAULT '{}'::jsonb,
#   created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

#   UNIQUE (dataset_id, key, country, frequency, transform)
# );

# -- releases = what you ingested (downloaded_at) and what "as-of" snapshot/vintage it represents (vintage_at)
# CREATE TABLE releases (
#   id            BIGSERIAL PRIMARY KEY,
#   dataset_id    BIGINT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,

#   -- keep release_time for backward compatibility; set equal to downloaded_at if you want
#   release_time  TIMESTAMPTZ NOT NULL,

#   downloaded_at TIMESTAMPTZ NOT NULL,   -- when your code downloaded/ingested it
#   vintage_at    TIMESTAMPTZ NOT NULL,   -- as-of time (vintage). For ALFRED = real vintage date. For others = snapshot time.

#   description   TEXT,
#   raw_path      TEXT,
#   content_hash  TEXT,
#   meta          JSONB NOT NULL DEFAULT '{}'::jsonb,

#   created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
# );

# CREATE TABLE observations (
#   id          BIGSERIAL PRIMARY KEY,
#   series_id   BIGINT NOT NULL REFERENCES series(id) ON DELETE CASCADE,
#   period_date DATE NOT NULL,
#   observed_at TIMESTAMPTZ NOT NULL,
#   value       DOUBLE PRECISION NOT NULL,
#   status      TEXT,
#   release_id  BIGINT REFERENCES releases(id) ON DELETE SET NULL,
#   meta        JSONB NOT NULL DEFAULT '{}'::jsonb,
#   created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

#   UNIQUE (series_id, period_date, observed_at)
# );

# CREATE TABLE ingestion_log (
#   id            BIGSERIAL PRIMARY KEY,
#   dataset_id    BIGINT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
#   run_time      TIMESTAMPTZ NOT NULL DEFAULT now(),
#   status        TEXT NOT NULL,
#   rows_inserted INTEGER NOT NULL DEFAULT 0,
#   rows_failed   INTEGER NOT NULL DEFAULT 0,
#   details       JSONB NOT NULL DEFAULT '{}'::jsonb
# );

# -- Useful indexes (minimal changes from yours)
# CREATE INDEX ix_series_dataset_id ON series(dataset_id);
# CREATE INDEX ix_observations_series_period ON observations(series_id, period_date);
# CREATE INDEX ix_observations_series_vintage ON observations(series_id, observed_at);
# CREATE INDEX ix_releases_dataset_vintage ON releases(dataset_id, vintage_at);
# CREATE INDEX ix_datasets_provider_id ON datasets(provider_id);
# """

# def main():
#     with engine.begin() as conn:
#         conn.execute(text(DDL))
#     print("Schema created")

# if __name__ == "__main__":
#     main()