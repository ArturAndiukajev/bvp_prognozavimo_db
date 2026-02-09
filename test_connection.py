from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"

engine = create_engine(DB_URL, future=True)

with engine.connect() as conn:
    print(conn.execute(text("select version();")).scalar_one())