"""
Šitas failas skirtas patikrinti esančius datasetus duombazėje.
"""
import pandas as pd
from sqlalchemy import create_engine

def main():
    engine = create_engine("postgresql+psycopg2://nowcast:nowcast@localhost:5432/nowcast_db")
    
    query = """
    SELECT 
        p.name AS provider,
        d.key AS dataset_key,
        COUNT(s.id) as num_series
    FROM datasets d
    JOIN providers p ON d.provider_id = p.id
    LEFT JOIN series s ON s.dataset_id = d.id
    GROUP BY p.name, d.key
    ORDER BY p.name, d.key;
    """
    
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("No datasets found in the database.")
    else:
        print("=== Available Datasets ===")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
