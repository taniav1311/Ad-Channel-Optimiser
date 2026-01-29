import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data_raw"

def load_marketing_data():
    print("Files in data_raw:")
    csv_files = list(RAW_DIR.glob("*.csv"))
    for file in csv_files:
        print(f"  - {file.name}")
    
    if not csv_files:
        print("ERROR: No CSV files found!")
        return
    
    # Load the FIRST CSV file automatically
    file_path = csv_files[0]
    print(f"\nLoading: {file_path.name}")
    
    df = pd.read_csv(file_path)
    
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn names:")
    print(list(df.columns))
    print(f"\nShape: {df.shape}")

if __name__ == "__main__":
    load_marketing_data()
