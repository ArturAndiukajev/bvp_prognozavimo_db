# from sqlalchemy import create_engine, text

# engine = create_engine("postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db")

# with engine.connect() as conn:
#     rows = conn.execute(text("""
#         select schemaname, tablename
#         from pg_catalog.pg_tables
#         where schemaname = 'public'
#         order by tablename
#     """)).fetchall()

# print("Tables found:", len(rows))
# for s, t in rows:
#     print(f"{s}.{t}")

from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db"
engine = create_engine(DB_URL, future=True)

with engine.connect() as conn:
    print("Sources:",
          conn.execute(text("select count(*) from sources")).scalar_one())

    print("Series:",
          conn.execute(text("select count(*) from series")).scalar_one())

    print("Observations:",
          conn.execute(text("select count(*) from observations")).scalar_one())
        