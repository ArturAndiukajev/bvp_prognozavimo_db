from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
engine = create_engine(DB_URL, future=True)

DDL = """
create table if not exists sources (
  id bigserial primary key,
  name text not null unique,
  base_url text,
  meta jsonb not null default '{}'::jsonb
);

create table if not exists series (
  id bigserial primary key,
  source_id bigint not null references sources(id),
  key text not null,
  country text,
  frequency text not null,
  transform text not null default 'LEVEL',
  unit text,
  name text,
  meta jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (source_id, key, country, frequency, transform)
);

create index if not exists idx_series_lookup
  on series (key, country, frequency, transform);

create table if not exists observations (
  series_id bigint not null references series(id),
  period_date date not null,
  observed_at timestamptz not null,
  value double precision,
  status text,
  meta jsonb not null default '{}'::jsonb,
  primary key (series_id, period_date, observed_at)
);

create index if not exists idx_obs_series_period
  on observations (series_id, period_date);

create index if not exists idx_obs_observed_at
  on observations (observed_at);

create table if not exists ingestion_log (
  id bigserial primary key,
  source_id bigint references sources(id),
  run_at timestamptz not null default now(),
  status text not null,
  rows_inserted bigint not null default 0,
  rows_failed bigint not null default 0,
  details jsonb not null default '{}'::jsonb
);
"""

with engine.begin() as conn:
    conn.execute(text(DDL))

print("Schema created")