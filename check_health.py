"""
check_health.py — Database health and freshness monitoring script.
Outputs per-source freshness, row counts, stale-detection, and
flags any source that hasn't been updated recently.
"""
import os
import logging
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("HealthCheck")

_DEFAULT_DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
DB_URL = os.environ.get("DB_URL", _DEFAULT_DB_URL)
engine = create_engine(
    DB_URL,
    future=True,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 10},
)

# Thresholds: how many hours before a source is considered stale
STALE_THRESHOLDS = {
    "yahoo_finance":  36,   # daily data — stale after 36h
    "fredmd":         200,  # monthly release
    "alfred":         200,  # infrequent vintages
    "eurostat":       200,  # quarterly / monthly Eurostat
    "google_trends":  200,  # monthly pulls
}
DEFAULT_STALE_HOURS = 200


def check_freshness(conn) -> list[dict]:
    """Return freshness info per provider."""
    rows = conn.execute(text("""
        SELECT
            p.name                                      AS source,
            d.key                                       AS dataset,
            count(DISTINCT s.id)                        AS series_count,
            count(o.id)                                 AS obs_count,
            max(o.created_at)                           AS last_obs_created,
            max(il.run_time)                            AS last_run_time,
            (count(CASE WHEN il.status NOT LIKE 'ok%' THEN 1 END))
                                                        AS error_runs
        FROM providers p
        JOIN datasets d ON d.provider_id = p.id
        LEFT JOIN series s   ON s.dataset_id = d.id
        LEFT JOIN observations o ON o.series_id = s.id
        LEFT JOIN ingestion_log il ON il.dataset_id = d.id
        GROUP BY p.name, d.key
        ORDER BY p.name, d.key
    """)).fetchall()
    return [dict(r._mapping) for r in rows]


def check_stale(freshness: list[dict]) -> list[dict]:
    """Flag sources whose last ingestion run is older than their threshold."""
    now = datetime.now(timezone.utc)
    stale = []
    for row in freshness:
        name = row["source"]
        last_run = row["last_run_time"]
        if last_run is None:
            stale.append({**row, "status": "NEVER_RUN"})
            continue
        # make tz-aware
        if last_run.tzinfo is None:
            last_run = last_run.replace(tzinfo=timezone.utc)
        hours_since = (now - last_run).total_seconds() / 3600
        threshold = STALE_THRESHOLDS.get(name, DEFAULT_STALE_HOURS)
        if hours_since > threshold:
            stale.append({**row, "status": f"STALE ({hours_since:.0f}h ago)"})
    return stale


def check_gaps(conn, provider_name: str) -> list[dict]:
    """
    Detect time gaps larger than 2× the typical interval for each series
    of the given provider. Only checks monthly (M) and quarterly (Q) series.
    """
    rows = conn.execute(text("""
        WITH ordered AS (
            SELECT
                o.series_id,
                s.key,
                o.period_date,
                LAG(o.period_date) OVER (
                    PARTITION BY o.series_id ORDER BY o.period_date
                ) AS prev_date
            FROM observations o
            JOIN series s ON s.id = o.series_id
            JOIN datasets d ON d.id = s.dataset_id
            JOIN providers p ON p.id = d.provider_id
            WHERE p.name = :src
              AND s.frequency IN ('M', 'Q')
              -- look at latest vintage only
              AND o.observed_at = (
                    SELECT MAX(o2.observed_at)
                    FROM observations o2
                    WHERE o2.series_id = o.series_id
              )
        )
        SELECT
            series_id,
            key,
            prev_date,
            period_date,
            (period_date - prev_date) AS gap_days
        FROM ordered
        WHERE prev_date IS NOT NULL
          AND (period_date - prev_date) > 95   -- > ~3 months = suspicious gap
        ORDER BY gap_days DESC
        LIMIT 20
    """), {"src": provider_name}).fetchall()
    return [dict(r._mapping) for r in rows]


def check_duplicate_values(conn) -> list[dict]:
    """
    Find series where more than 50% of values are identical consecutive values
    (flat-line detection), which may indicate stale or bad data.
    """
    rows = conn.execute(text("""
        WITH ranked AS (
            SELECT
                o.series_id,
                s.key,
                o.period_date,
                o.value,
                LAG(o.value) OVER (
                    PARTITION BY o.series_id ORDER BY o.period_date
                ) AS prev_value
            FROM observations o
            JOIN series s ON s.id = o.series_id
            -- only latest vintage
            WHERE o.observed_at = (
                SELECT MAX(o2.observed_at) FROM observations o2
                WHERE o2.series_id = o.series_id
            )
        )
        SELECT
            series_id,
            key,
            COUNT(*)                                            AS total_obs,
            SUM(CASE WHEN value = prev_value THEN 1 ELSE 0 END) AS flat_count,
            ROUND(
                100.0 * SUM(CASE WHEN value = prev_value THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0), 1
            )                                                   AS flat_pct
        FROM ranked
        GROUP BY series_id, key
        HAVING COUNT(*) > 10
           AND (
               100.0 * SUM(CASE WHEN value = prev_value THEN 1 ELSE 0 END)
               / NULLIF(COUNT(*), 0)
           ) > 50
        ORDER BY flat_pct DESC
        LIMIT 20
    """)).fetchall()
    return [dict(r._mapping) for r in rows]


def main():
    logger.info("=== DB Health Check ===")

    with engine.connect() as conn:
        freshness = check_freshness(conn)

    # --- Freshness table ---
    now = datetime.now(timezone.utc)
    print(f"\n{'Provider':<15} {'Dataset':<30} {'Series':>8} {'Obs':>12} {'Last Run':<25} {'Hours Ago':>10} {'Errors':>7}")
    print("-" * 115)
    for row in freshness:
        last_run = row["last_run_time"]
        if last_run and last_run.tzinfo is None:
            last_run = last_run.replace(tzinfo=timezone.utc)
        hours_ago = f"{(now - last_run).total_seconds()/3600:.0f}" if last_run else "N/A"
        last_run_str = last_run.strftime("%Y-%m-%d %H:%M UTC") if last_run else "Never"
        print(
            f"{row['source']:<15} {row['dataset']:<30} {row['series_count']:>8} {row['obs_count']:>12} "
            f"{last_run_str:<25} {hours_ago:>10} {row['error_runs']:>7}"
        )

    # --- Stale sources ---
    stale = check_stale(freshness)
    if stale:
        print(f"\n[!] STALE SOURCES ({len(stale)}):")
        for s in stale:
            print(f"  * {s['source']}: {s['status']}")
    else:
        print("\n[OK] All sources are fresh.")

    # --- Gap detection (per source) ---
    with engine.connect() as conn:
        for src in ["eurostat", "fredmd", "alfred"]:
            gaps = check_gaps(conn, src)
            if gaps:
                print(f"\n[!] Gaps detected in [{src}]:")
                for g in gaps:
                    print(f"  * {g['key']}: {g['prev_date']} -> {g['period_date']} ({g['gap_days']} days)")
            else:
                logger.info(f"No gaps detected in [{src}]")

    # --- Flat-line detection ---
    with engine.connect() as conn:
        flatlines = check_duplicate_values(conn)
    if flatlines:
        print(f"\n[!] Potentially flat series ({len(flatlines)}):")
        for f in flatlines:
            print(f"  * {f['key']}: {f['flat_pct']}% flat ({f['flat_count']}/{f['total_obs']} obs)")
    else:
        print("\n[OK] No flat-line series detected.")

    logger.info("=== Health Check Complete ===")


if __name__ == "__main__":
    main()
