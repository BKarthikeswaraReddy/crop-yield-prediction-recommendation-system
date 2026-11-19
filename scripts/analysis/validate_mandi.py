
from pathlib import Path
import pandas as pd

folder = Path("data/mandi_prices")

print("=" * 65)
print("MANDI PRICE DATASET VALIDATION")
print("=" * 65)

files = sorted(folder.glob("*.parquet"))

print("\nTOTAL FILES:", len(files))

for file in files:
    print(f"\nYEAR FILE: {file.name}")

    try:
        df = pd.read_parquet(file)

        print("Records:", len(df))
        print("Columns:", list(df.columns))

        if "Arrival_Date" in df.columns:
            dates = pd.to_datetime(
                df["Arrival_Date"], errors="coerce"
            )
            print("Start date:", dates.min())
            print("End date:", dates.max())

        print("Missing values:")
        print(df.isnull().sum().to_dict())

        if "Modal_Price" in df.columns:
            print("Zero modal prices:", (df["Modal_Price"] == 0).sum())
            print("Negative modal prices:", (df["Modal_Price"] < 0).sum())

    except Exception as e:
        print("ERROR:", e)

print("\nMANDI VALIDATION COMPLETED")
print("=" * 65)
