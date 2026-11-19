
import pandas as pd
from pathlib import Path

file_path = Path("data/India Agriculture Crop Production.csv")

print("=" * 65)
print("CROP PRODUCTION DATASET - DETAILED VALIDATION")
print("=" * 65)

df = pd.read_csv(file_path)

print("\n1. DATASET SHAPE")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n2. COLUMN DATA TYPES")
print(df.dtypes)

print("\n3. MISSING VALUES")
print(df.isnull().sum())

print("\n4. DUPLICATE ROWS")
print(df.duplicated().sum())

print("\n5. UNIQUE COUNTS")
for col in ["State", "District", "Crop", "Year", "Season",
            "Area Units", "Production Units"]:
    if col in df.columns:
        print(f"{col}: {df[col].nunique()}")

print("\n6. YEAR COVERAGE")
print("Minimum year:", df["Year"].min())
print("Maximum year:", df["Year"].max())
print("Years available:")
print(sorted(df["Year"].dropna().unique().tolist()))

print("\n7. PRODUCTION UNIT COUNTS")
print(df["Production Units"].value_counts(dropna=False))

print("\n8. AREA UNIT COUNTS")
print(df["Area Units"].value_counts(dropna=False))

print("\n9. INVALID OR ZERO VALUES")
for col in ["Area", "Production", "Yield"]:
    if col in df.columns:
        print(f"\n{col}")
        print("Zero:", (df[col] == 0).sum())
        print("Negative:", (df[col] < 0).sum())
        print("Missing:", df[col].isna().sum())

print("\n10. TOP 10 STATES")
print(df["State"].value_counts().head(10))

print("\n11. TOP 10 CROPS")
print(df["Crop"].value_counts().head(10))

print("\n12. VALIDATION COMPLETED")
print("=" * 65)
