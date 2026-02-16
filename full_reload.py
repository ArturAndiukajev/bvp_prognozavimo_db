import create_schema
from scripts import fredmd
from scripts import load_alfred
from scripts import load_eurostat
from scripts import load_google_trends
from scripts import load_financials

def main():
    print("=== Full Reload Process Started ===")
    
    print("Recreating schema...")
    create_schema.main()

    print("Loading data...")
    try:
        fredmd.main()
        load_alfred.main()
        load_eurostat.main()
        load_google_trends.main()
        load_financials.main()
    except Exception as e:
        print(f"Error during data loading: {e}")
        return

    print("=== Full Reload Complete ===")

if __name__ == "__main__":
    main()
