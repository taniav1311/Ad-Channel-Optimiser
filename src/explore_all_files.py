import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data_raw"

def explore_all():
    csv_files = list(RAW_DIR.glob("*.csv"))
    
    print("=== ALL FILES OVERVIEW ===\n")
    
    for file_path in csv_files:
        print(f"\n📄 {file_path.name}")
        df = pd.read_csv(file_path)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample values:")
        for col in df.columns[:4]:  # First 4 columns
            print(f"    {col}: {df[col].dropna().unique()[:3]}")

explore_all()
