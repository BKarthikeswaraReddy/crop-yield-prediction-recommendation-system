
import pandas as pd
from pathlib import Path

file_path = Path("data/Cost_of_Cultivation/cost-of-cultivation.csv")

df = pd.read_csv(file_path)

print("=" * 65)
print("COST OF CULTIVATION - DETAILED VALIDATION")
print("=" * 65)

print("\n1. DATASET SHAPE")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n2. MISSING VALUES")
print(df.isnull().sum().to_string())

print("\n3. DUPLICATE ROWS")
print(df.duplicated().sum())

print("\n4. YEAR COVERAGE")
print("Years:", sorted(df["year"].dropna().unique().tolist()))

print("\n5. STATE COVERAGE")
print("Unique states:", df["state_name"].nunique())
print(sorted(df["state_name"].dropna().unique().tolist()))

print("\n6. CROP COVERAGE")
print("Unique crops:", df["crop_name"].nunique())
print(sorted(df["crop_name"].dropna().unique().tolist()))

print("\n7. COST COLUMNS SUMMARY")
cost_cols = [c for c in df.columns if "cost" in c.lower()]

for col in cost_cols:
    print(
        col,
        "| Missing:", df[col].isna().sum(),
        "| Minimum:", df[col].min(),
        "| Maximum:", df[col].max()
    )

print("\n8. VALIDATION COMPLETED")
print("=" * 65)
