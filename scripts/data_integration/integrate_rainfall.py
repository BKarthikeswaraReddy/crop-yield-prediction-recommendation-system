"""
Rainfall Data Integration Pipeline — Stage 1 Weather Preparation
Crop Production Prediction and Farmer Decision Support System

Consolidates district-wise and state-wise rainfall CSV files into standardized,
deduplicated Parquet datasets with full data quality auditing.

Sources:
  District-level:
    - 8 yearly UUID CSV files (2018-2025): State, District, Date, Year, Month, Avg_rainfall, Agency_name
    - rainfall_districtwise_daily_imd.csv (2026): State, District, Date, Daily Actual, ...
  State-level:
    - rainfall_statewise_daily_imd.csv: State, Date, Daily Actual, ...

Outputs:
    data/weather/rainfall_districtwise_integrated.parquet
    data/weather/rainfall_statewise_integrated.parquet
    data/weather/rainfall_integration_summary.csv

Rules:
    - Negative rainfall values → null (never zero).
    - Genuine zero-rainfall observations → preserved.
    - Missing rainfall → preserved as null (never filled with invented values).
    - Exact duplicate rows → removed (specifically the duplicated 2025 records).
    - Records with missing rainfall → kept (not discarded).
    - District-level and state-level rainfall → never combined.
    - Original source files → never modified.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
WEATHER_DIR = DATA_DIR / "weather"

DISTRICT_OUTPUT = WEATHER_DIR / "rainfall_districtwise_integrated.parquet"
STATEWISE_OUTPUT = WEATHER_DIR / "rainfall_statewise_integrated.parquet"
SUMMARY_FILE = WEATHER_DIR / "rainfall_integration_summary.csv"

# Temporary files for safe writing
DISTRICT_TEMP = WEATHER_DIR / "rainfall_districtwise_integrated.tmp.parquet"
STATEWISE_TEMP = WEATHER_DIR / "rainfall_statewise_integrated.tmp.parquet"

# -----------------------------------------------------------------------------
# SCHEMAS
# -----------------------------------------------------------------------------

DISTRICT_SCHEMA = pa.schema([
    ("State", pa.string()),
    ("District", pa.string()),
    ("Date", pa.date32()),
    ("Year", pa.int32()),
    ("Month", pa.int32()),
    ("Rainfall", pa.float64()),
    ("Agency_name", pa.string()),
    ("Source_file", pa.string()),
])

STATEWISE_SCHEMA = pa.schema([
    ("State", pa.string()),
    ("Date", pa.date32()),
    ("Year", pa.int32()),
    ("Month", pa.int32()),
    ("Rainfall", pa.float64()),
    ("Source_file", pa.string()),
])


# -----------------------------------------------------------------------------
# DISTRICT-LEVEL INTEGRATION
# -----------------------------------------------------------------------------

def integrate_district_rainfall() -> List[Dict]:
    """Integrate all district-level rainfall files into one Parquet dataset.

    Returns list of per-source summary dicts.
    """
    print("\n" + "=" * 85)
    print("DISTRICT-LEVEL RAINFALL INTEGRATION")
    print("=" * 85)

    summaries: List[Dict] = []

    # --- Discover UUID yearly files (2018-2025) ---
    uuid_files = sorted(WEATHER_DIR.glob("6c05cd1b-*.csv"))
    if len(uuid_files) != 8:
        raise ValueError(f"Expected 8 yearly UUID CSV files, found {len(uuid_files)}")

    # --- Discover IMD district file (2026) ---
    imd_district_file = WEATHER_DIR / "rainfall_districtwise_daily_imd.csv"
    if not imd_district_file.exists():
        raise FileNotFoundError(f"Missing: {imd_district_file}")

    all_sources = list(uuid_files) + [imd_district_file]
    print(f"Found {len(all_sources)} district-level rainfall source files.\n")

    # Clean up stale temp files
    if DISTRICT_TEMP.exists():
        DISTRICT_TEMP.unlink()

    writer: pq.ParquetWriter = None

    try:
        writer = pq.ParquetWriter(
            DISTRICT_TEMP, DISTRICT_SCHEMA, compression="snappy"
        )

        for file_path in all_sources:
            print(f"Processing: {file_path.name}")
            df = pd.read_csv(file_path)
            source_rows = len(df)

            # --- Standardize columns ---
            if "Avg_rainfall" in df.columns:
                # UUID yearly files: State, District, Date, Year, Month, Avg_rainfall, Agency_name
                std = pd.DataFrame()
                std["State"] = df["State"].astype(str).str.strip()
                std["District"] = df["District"].astype(str).str.strip()

                parsed_dates = pd.to_datetime(df["Date"], errors="coerce")
                std["Date"] = parsed_dates.dt.date
                std["Year"] = parsed_dates.dt.year.astype("Int32")
                std["Month"] = parsed_dates.dt.month.astype("Int32")

                rainfall = pd.to_numeric(df["Avg_rainfall"], errors="coerce")
                std["Rainfall"] = rainfall.astype(float)
                std["Agency_name"] = df["Agency_name"].astype(str).str.strip()

            elif "Daily Actual" in df.columns:
                # IMD district file: State, District, Date, Daily Actual, ...
                std = pd.DataFrame()
                std["State"] = df["State"].astype(str).str.strip()
                std["District"] = df["District"].astype(str).str.strip()

                parsed_dates = pd.to_datetime(df["Date"], errors="coerce")
                std["Date"] = parsed_dates.dt.date
                std["Year"] = parsed_dates.dt.year.astype("Int32")
                std["Month"] = parsed_dates.dt.month.astype("Int32")

                rainfall = pd.to_numeric(df["Daily Actual"], errors="coerce")
                std["Rainfall"] = rainfall.astype(float)
                std["Agency_name"] = "IMD"  # Official IMD source

            else:
                raise ValueError(f"Unrecognized column layout in {file_path.name}: {list(df.columns)}")

            std["Source_file"] = file_path.name

            # --- Exact duplicate removal ---
            before_dedup = len(std)
            std = std.drop_duplicates().copy()
            exact_duplicates_removed = before_dedup - len(std)

            # --- Negative rainfall → null (never zero) ---
            negative_mask = std["Rainfall"] < 0
            negative_count = int(negative_mask.sum())
            std.loc[negative_mask, "Rainfall"] = np.nan

            # --- Audit statistics ---
            after_rows = len(std)
            null_rainfall = int(std["Rainfall"].isna().sum())
            zero_rainfall = int((std["Rainfall"] == 0).sum())
            positive_rainfall = int((std["Rainfall"] > 0).sum())
            date_range_min = std["Date"].min()
            date_range_max = std["Date"].max()
            years_present = sorted(std["Year"].dropna().unique().tolist())

            # --- Stream to Parquet ---
            table = pa.Table.from_pandas(std, schema=DISTRICT_SCHEMA, preserve_index=False)
            writer.write_table(table)

            summary = {
                "dataset": "district",
                "source_file": file_path.name,
                "source_rows": source_rows,
                "exact_duplicates_removed": exact_duplicates_removed,
                "negative_rainfall_nullified": negative_count,
                "output_rows": after_rows,
                "null_rainfall": null_rainfall,
                "zero_rainfall": zero_rainfall,
                "positive_rainfall": positive_rainfall,
                "date_min": str(date_range_min),
                "date_max": str(date_range_max),
                "years": "; ".join(str(y) for y in years_present),
            }
            summaries.append(summary)

            print(
                f"  Source={source_rows:,} | DupsRemoved={exact_duplicates_removed:,} | "
                f"NegNulled={negative_count} | Output={after_rows:,} | "
                f"Dates={date_range_min} to {date_range_max}"
            )

            del df, std, table

        writer.close()
        writer = None

        # Atomic promotion
        os.replace(DISTRICT_TEMP, DISTRICT_OUTPUT)
        print(f"\n[OK] District rainfall saved: {DISTRICT_OUTPUT}")

    except Exception as exc:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if DISTRICT_TEMP.exists():
            DISTRICT_TEMP.unlink()
        raise exc

    return summaries


# -----------------------------------------------------------------------------
# STATE-LEVEL INTEGRATION
# -----------------------------------------------------------------------------

def integrate_statewise_rainfall() -> List[Dict]:
    """Integrate state-level rainfall CSV into a standardized Parquet dataset.

    Returns list of per-source summary dicts.
    """
    print("\n" + "=" * 85)
    print("STATE-LEVEL RAINFALL INTEGRATION")
    print("=" * 85)

    summaries: List[Dict] = []

    state_file = WEATHER_DIR / "rainfall_statewise_daily_imd.csv"
    if not state_file.exists():
        raise FileNotFoundError(f"Missing: {state_file}")

    # Clean up stale temp files
    if STATEWISE_TEMP.exists():
        STATEWISE_TEMP.unlink()

    print(f"Processing: {state_file.name}")
    df = pd.read_csv(state_file)
    source_rows = len(df)

    # Standardize columns
    std = pd.DataFrame()
    std["State"] = df["State"].astype(str).str.strip()

    parsed_dates = pd.to_datetime(df["Date"], errors="coerce")
    std["Date"] = parsed_dates.dt.date
    std["Year"] = parsed_dates.dt.year.astype("Int32")
    std["Month"] = parsed_dates.dt.month.astype("Int32")

    rainfall = pd.to_numeric(df["Daily Actual"], errors="coerce")
    std["Rainfall"] = rainfall.astype(float)
    std["Source_file"] = state_file.name

    # Exact duplicate removal
    before_dedup = len(std)
    std = std.drop_duplicates().copy()
    exact_duplicates_removed = before_dedup - len(std)

    # Negative rainfall → null
    negative_mask = std["Rainfall"] < 0
    negative_count = int(negative_mask.sum())
    std.loc[negative_mask, "Rainfall"] = np.nan

    # Audit
    after_rows = len(std)
    null_rainfall = int(std["Rainfall"].isna().sum())
    zero_rainfall = int((std["Rainfall"] == 0).sum())
    positive_rainfall = int((std["Rainfall"] > 0).sum())
    date_range_min = std["Date"].min()
    date_range_max = std["Date"].max()
    years_present = sorted(std["Year"].dropna().unique().tolist())

    print(
        f"  Source={source_rows:,} | DupsRemoved={exact_duplicates_removed:,} | "
        f"NegNulled={negative_count} | Output={after_rows:,} | "
        f"Dates={date_range_min} to {date_range_max}"
    )

    # Write with safe temp file
    table = pa.Table.from_pandas(std, schema=STATEWISE_SCHEMA, preserve_index=False)
    pq.write_table(table, STATEWISE_TEMP, compression="snappy")
    os.replace(STATEWISE_TEMP, STATEWISE_OUTPUT)
    print(f"\n[OK] State rainfall saved: {STATEWISE_OUTPUT}")

    summaries.append({
        "dataset": "statewise",
        "source_file": state_file.name,
        "source_rows": source_rows,
        "exact_duplicates_removed": exact_duplicates_removed,
        "negative_rainfall_nullified": negative_count,
        "output_rows": after_rows,
        "null_rainfall": null_rainfall,
        "zero_rainfall": zero_rainfall,
        "positive_rainfall": positive_rainfall,
        "date_min": str(date_range_min),
        "date_max": str(date_range_max),
        "years": "; ".join(str(y) for y in years_present),
    })

    del df, std, table
    return summaries


# -----------------------------------------------------------------------------
# VERIFICATION
# -----------------------------------------------------------------------------

def verify_outputs():
    """Read back outputs and print verification statistics."""
    print("\n" + "=" * 85)
    print("OUTPUT VERIFICATION")
    print("=" * 85)

    # District verification
    print(f"\n--- District Parquet: {DISTRICT_OUTPUT.name} ---")
    dt = pq.read_table(DISTRICT_OUTPUT)
    ddf = dt.to_pandas()
    print(f"Schema fields:   {[f.name for f in dt.schema]}")
    print(f"Total rows:      {len(ddf):,}")
    print(f"Unique states:   {ddf['State'].nunique()}")
    print(f"Unique districts:{ddf['District'].nunique()}")
    print(f"Date range:      {ddf['Date'].min()} to {ddf['Date'].max()}")
    print(f"Years:           {sorted(ddf['Year'].dropna().unique().tolist())}")
    print(f"Agencies:        {ddf['Agency_name'].value_counts().to_dict()}")
    print(f"Null rainfall:   {ddf['Rainfall'].isna().sum():,}")
    print(f"Zero rainfall:   {(ddf['Rainfall'] == 0).sum():,}")
    print(f"Positive:        {(ddf['Rainfall'] > 0).sum():,}")
    print(f"Exact dups left: {ddf.duplicated().sum()}")
    del dt, ddf

    # Statewise verification
    print(f"\n--- Statewise Parquet: {STATEWISE_OUTPUT.name} ---")
    st = pq.read_table(STATEWISE_OUTPUT)
    sdf = st.to_pandas()
    print(f"Schema fields:   {[f.name for f in st.schema]}")
    print(f"Total rows:      {len(sdf):,}")
    print(f"Unique states:   {sdf['State'].nunique()}")
    print(f"Date range:      {sdf['Date'].min()} to {sdf['Date'].max()}")
    print(f"Years:           {sorted(sdf['Year'].dropna().unique().tolist())}")
    print(f"Null rainfall:   {sdf['Rainfall'].isna().sum():,}")
    print(f"Zero rainfall:   {(sdf['Rainfall'] == 0).sum():,}")
    print(f"Positive:        {(sdf['Rainfall'] > 0).sum():,}")
    print(f"Exact dups left: {sdf.duplicated().sum()}")
    del st, sdf


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    print("=" * 85)
    print("RAINFALL DATA INTEGRATION PIPELINE — STAGE 1")
    print("Crop Production Prediction and Farmer Decision Support System")
    print("=" * 85)

    # 1. District-level integration
    district_summaries = integrate_district_rainfall()

    # 2. State-level integration (kept entirely separate)
    state_summaries = integrate_statewise_rainfall()

    # 3. Save combined summary CSV
    all_summaries = district_summaries + state_summaries
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    print(f"\n[OK] Integration summary saved: {SUMMARY_FILE}")

    # 4. Verification
    verify_outputs()

    # 5. Final totals
    print("\n" + "=" * 85)
    print("RAINFALL INTEGRATION COMPLETED SUCCESSFULLY")
    print("=" * 85)
    total_source = sum(s["source_rows"] for s in all_summaries)
    total_dups = sum(s["exact_duplicates_removed"] for s in all_summaries)
    total_neg = sum(s["negative_rainfall_nullified"] for s in all_summaries)
    total_output = sum(s["output_rows"] for s in all_summaries)
    print(f"Total source records:            {total_source:,}")
    print(f"Exact duplicates removed:        {total_dups:,}")
    print(f"Negative rainfall → null:        {total_neg:,}")
    print(f"Total output records:            {total_output:,}")
    print(f"District Parquet:                {DISTRICT_OUTPUT}")
    print(f"State Parquet:                   {STATEWISE_OUTPUT}")
    print(f"Summary CSV:                     {SUMMARY_FILE}")
    print("=" * 85)


if __name__ == "__main__":
    main()

