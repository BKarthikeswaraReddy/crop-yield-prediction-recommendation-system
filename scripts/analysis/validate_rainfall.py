
from pathlib import Path
import pandas as pd

folder = Path("data/weather")

files = list(folder.glob("*.csv"))

print("=" * 70)
print("RAINFALL DATASET VALIDATION")
print("=" * 70)

for file in sorted(files):
    print("\nFILE:", file.name)

    try:
        df = pd.read_csv(file)

        print("Rows:", len(df))
        print("Columns:", list(df.columns))
        print("Missing values:", df.isnull().sum().sum())
        print("Duplicate rows:", df.duplicated().sum())

        if "Date" in df.columns:
            dates = pd.to_datetime(df["Date"], errors="coerce")
            print("Start date:", dates.min())
            print("End date:", dates.max())

        if "Year" in df.columns:
            print("Years:", sorted(df["Year"].dropna().unique().tolist()))

        if "State" in df.columns:
            print("Unique states:", df["State"].nunique())

        if "District" in df.columns:
            print("Unique districts:", df["District"].nunique())

        if "Avg_rainfall" in df.columns:
            print("Rainfall missing:", df["Avg_rainfall"].isna().sum())
            print("Negative rainfall:", (df["Avg_rainfall"] < 0).sum())

        if "Daily Actual" in df.columns:
            print("Daily Actual missing:", df["Daily Actual"].isna().sum())

    except Exception as error:
        print("ERROR:", error)

print("\nRAINFALL VALIDATION COMPLETED")
print("=" * 70)
