import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

BASE_DIR = Path("crop_Production_prediction_new/crop-production-predictor")
DATA_DIR = BASE_DIR / "data"

print("=" * 80)
print("CHECK 1: PRODUCTION DATASET VERIFICATION & 331,654 RECORD EXPLANATION")
print("=" * 80)
prod_csv = DATA_DIR / "India Agriculture Crop Production.csv"
print(f"Dataset path: {prod_csv.resolve()}")
print(f"Exists: {prod_csv.exists()}")

df_prod = pd.read_csv(prod_csv)
print(f"Total raw rows in CSV: {len(df_prod):,}")
print(f"Columns: {list(df_prod.columns)}")
print(f"Missing values by column in raw CSV:\n{df_prod.isna().sum()}")

print("\nProduction Units breakdown:")
units_dist = df_prod["Production Units"].value_counts(dropna=False)
print(units_dist)

# Check 331,654 figure vs other subsets
# Non-null Production rows:
non_null_prod = df_prod["Production"].notna().sum()
print(f"Rows with non-null Production: {non_null_prod:,}")

# Rows with non-null Production and valid units:
tonnes_mask = df_prod["Production Units"].astype(str).str.strip().str.lower() == "tonnes"
print(f"Rows where Production Units == 'Tonnes' (case-insensitive): {tonnes_mask.sum():,}")

# What makes 331,654? Let's check non-null or specific subsets:
# E.g. raw minus missing production?
# Total raw (345,417 or 345,338?) Let's see:
print(f"Exact total rows: {len(df_prod)}")

# Let's check cost integrated dataset as well:
cost_csv = DATA_DIR / "production_cost_integrated.csv"
if cost_csv.exists():
    df_cost = pd.read_csv(cost_csv, nrows=10)
    print(f"\nproduction_cost_integrated.csv columns: {list(df_cost.columns)}")

print("\n" + "=" * 80)
print("CHECK 2: TEMPERATURE DATASETS INSPECTION")
print("=" * 80)
temp_grid = DATA_DIR / "weather" / "temperature_gridwise_integrated.parquet"
temp_month = DATA_DIR / "weather" / "temperature_monthly_summary.parquet"

print(f"temperature_gridwise_integrated.parquet exists: {temp_grid.exists()}")
if temp_grid.exists():
    pf_grid = pq.ParquetFile(temp_grid)
    print(f"  Rows: {pf_grid.metadata.num_rows:,}")
    print(f"  Columns: {pf_grid.metadata.num_columns}")
    print(f"  Schema: {pf_grid.schema.names}")
    # Inspect sample & dates
    df_tg_sample = pd.read_parquet(temp_grid)
    print(f"  Sample shape: {df_tg_sample.shape}")
    print(f"  Missing values:\n{df_tg_sample.isna().sum()}")
    date_col = [c for c in df_tg_sample.columns if "date" in c.lower() or "year" in c.lower()]
    print(f"  Date/Year columns: {date_col}")
    for dc in date_col:
        print(f"    {dc} range: {df_tg_sample[dc].min()} to {df_tg_sample[dc].max()}")

print(f"\ntemperature_monthly_summary.parquet exists: {temp_month.exists()}")
if temp_month.exists():
    pf_month = pq.ParquetFile(temp_month)
    print(f"  Rows: {pf_month.metadata.num_rows:,}")
    print(f"  Columns: {pf_month.metadata.num_columns}")
    print(f"  Schema: {pf_month.schema.names}")
    df_tm = pd.read_parquet(temp_month)
    print(f"  Missing values:\n{df_tm.isna().sum()}")
    date_cols_m = [c for c in df_tm.columns if "date" in c.lower() or "year" in c.lower() or "month" in c.lower()]
    print(f"  Date/Year/Month columns: {date_cols_m}")
    for dc in date_cols_m:
        print(f"    {dc} range: {df_tm[dc].min()} to {df_tm[dc].max()}")

print("\n" + "=" * 80)
print("CHECK 3: ALL INPUT DATASETS AND MATCHING KEYS")
print("=" * 80)
rainfall_dist = DATA_DIR / "weather" / "rainfall_districtwise_integrated.parquet"
mandi_file = DATA_DIR / "mandi_prices_integrated.parquet"

print(f"rainfall_districtwise_integrated.parquet exists: {rainfall_dist.exists()}")
if rainfall_dist.exists():
    pf_rf = pq.ParquetFile(rainfall_dist)
    print(f"  Rainfall rows: {pf_rf.metadata.num_rows:,}, columns: {pf_rf.schema.names}")

print(f"mandi_prices_integrated.parquet exists: {mandi_file.exists()}")
if mandi_file.exists():
    pf_mandi = pq.ParquetFile(mandi_file)
    print(f"  Mandi rows: {pf_mandi.metadata.num_rows:,}, columns: {pf_mandi.schema.names}")

print("\n" + "=" * 80)
print("CHECK 4 & 5: TARGET LEAKAGE, CHRONOLOGICAL SPLIT & TARGET VALIDITY")
print("=" * 80)
# Check production & yield distributions
# Filter tonnes
df_tonnes = df_prod[tonnes_mask].copy()
df_tonnes["Area"] = pd.to_numeric(df_tonnes["Area"], errors="coerce")
df_tonnes["Production"] = pd.to_numeric(df_tonnes["Production"], errors="coerce")
df_tonnes = df_tonnes.dropna(subset=["State", "District", "Crop", "Year", "Season", "Area", "Production"])
df_tonnes = df_tonnes[(df_tonnes["Area"] > 0) & (df_tonnes["Production"] >= 0)]
df_tonnes["Yield_Calculated"] = df_tonnes["Production"] / df_tonnes["Area"]

print(f"Valid clean records in Tonnes: {len(df_tonnes):,}")
print("Production target stats:")
print(df_tonnes["Production"].describe())

print("\nYield target stats:")
print(df_tonnes["Yield_Calculated"].describe())

df_tonnes["Year_Num"] = df_tonnes["Year"].astype(str).str.extract(r"(\d{4})").astype(float)
print(f"\nYear_Num range: {df_tonnes['Year_Num'].min()} to {df_tonnes['Year_Num'].max()}")
print("Year distribution:")
print(df_tonnes["Year_Num"].value_counts().sort_index())

