from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
engine = create_engine(DB_URL, future=True)

DDL = """
-- Clean drop (order matters with FKs, CASCADE simplifies)
DROP TABLE IF EXISTS ingestion_log CASCADE;
DROP TABLE IF EXISTS observations CASCADE;
DROP TABLE IF EXISTS releases CASCADE;
DROP TABLE IF EXISTS series CASCADE;
DROP TABLE IF EXISTS sources CASCADE;

CREATE TABLE sources (
  id          BIGSERIAL PRIMARY KEY,
  name        TEXT NOT NULL UNIQUE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  meta JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE series (
  id          BIGSERIAL PRIMARY KEY,
  source_id   BIGINT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  key         TEXT NOT NULL,
  country     TEXT NOT NULL,
  frequency   TEXT NOT NULL,       -- e.g. 'D','W','M','Q','A'
  transform   TEXT NOT NULL,       -- e.g. 'LEVEL','LOG','DIFF' etc (your label)
  unit        TEXT,
  name        TEXT,
  meta        JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE (source_id, key, country, frequency, transform)
);

-- releases = what you ingested (downloaded_at) and what "as-of" snapshot/vintage it represents (vintage_at)
CREATE TABLE releases (
  id            BIGSERIAL PRIMARY KEY,
  source_id     BIGINT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,

  -- keep release_time for backward compatibility; set equal to downloaded_at if you want
  release_time  TIMESTAMPTZ NOT NULL,

  downloaded_at TIMESTAMPTZ NOT NULL,   -- when your code downloaded/ingested it
  vintage_at    TIMESTAMPTZ NOT NULL,   -- as-of time (vintage). For ALFRED = real vintage date. For others = snapshot time.

  description   TEXT,
  raw_path      TEXT,                  -- path to raw file you saved (optional)
  content_hash  TEXT,                  -- sha256 of raw file (optional)
  meta          JSONB NOT NULL DEFAULT '{}'::jsonb,

  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE observations (
  id          BIGSERIAL PRIMARY KEY,
  series_id   BIGINT NOT NULL REFERENCES series(id) ON DELETE CASCADE,
  period_date DATE NOT NULL,              -- period the value refers to
  observed_at TIMESTAMPTZ NOT NULL,       -- vintage/as-of time
  value       DOUBLE PRECISION NOT NULL,
  status      TEXT,
  release_id  BIGINT REFERENCES releases(id) ON DELETE SET NULL,
  meta        JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE (series_id, period_date, observed_at)
);

CREATE TABLE ingestion_log (
  id            BIGSERIAL PRIMARY KEY,
  source_id     BIGINT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  run_time      TIMESTAMPTZ NOT NULL DEFAULT now(),
  status        TEXT NOT NULL,
  rows_inserted INTEGER NOT NULL DEFAULT 0,
  rows_failed   INTEGER NOT NULL DEFAULT 0,
  details       JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TEMP TABLE observations_staging (
    series_id   BIGINT,
    period_date DATE,
    observed_at TIMESTAMPTZ,
    value       DOUBLE PRECISION,
    release_id  BIGINT
) ON COMMIT DROP;

-- Useful indexes
CREATE INDEX ix_series_source_id ON series(source_id);
CREATE INDEX ix_observations_series_period ON observations(series_id, period_date);
CREATE INDEX ix_observations_series_vintage ON observations(series_id, observed_at);
CREATE INDEX ix_releases_source_vintage ON releases(source_id, vintage_at);
"""

def main():
    with engine.begin() as conn:
        conn.execute(text(DDL))
    print("Schema created")

if __name__ == "__main__":
    main()
#