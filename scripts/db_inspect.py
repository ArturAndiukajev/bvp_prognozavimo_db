from scripts import db_helpers
from sqlalchemy import text

engine = db_helpers.get_engine()
with engine.connect() as conn:
    res = conn.execute(text("""
        SELECT p.name, d.key as d_key, s.key as s_key, s.id, s.name as s_name
        FROM series s
        JOIN datasets d ON d.id = s.dataset_id
        JOIN providers p ON p.id = d.provider_id
        WHERE d.key ilike '%GDPC1%' OR s.key ilike '%GDPC1%'
    """))
    for row in res.fetchall():
        print(dict(row._mapping))
