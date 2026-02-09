from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
)

with engine.connect() as conn:
    result = conn.execute(text("""
        select s.key, o.period_date, o.value
        from observations o
        join series s on s.id=o.series_id
        limit 10
    """))

    for row in result:
        print(row)
