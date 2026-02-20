from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
)

with engine.connect() as conn:
    result = conn.execute(text("""
        select count(*)
        from observations o
        join series s on s.id=o.series_id
        join datasets d on d.id=s.dataset_id
        join providers p on p.id=d.provider_id
        where p.name='fredmd'
    """))

    print(result.scalar())

