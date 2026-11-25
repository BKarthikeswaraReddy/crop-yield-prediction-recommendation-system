"""
Rainfall Data Integration Pipeline — Stage 1 Weather Preparation
Machine Learning Based Crop Yield Prediction & Recommendation System

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
WEATHER_DIR = DATA_DIR / "weather"

DISTRICT_OUTPUT = WEATHER_DIR / "rainfall_districtwise_integrated.parquet"
STATEWISE_OUTPUT = WEATHER_DIR / "rainfall_statewise_integrated.parquet"
SEASON_SUMMARY_OUTPUT = WEATHER_DIR / "rainfall_district_season_summary.parquet"
SUMMARY_FILE = WEATHER_DIR / "rainfall_integration_summary.csv"

# Temporary files for safe writing
DISTRICT_TEMP = WEATHER_DIR / "rainfall_districtwise_integrated.tmp.parquet"
STATEWISE_TEMP = WEATHER_DIR / "rainfall_statewise_integrated.tmp.parquet"
SEASON_SUMMARY_TEMP = WEATHER_DIR / "rainfall_district_season_summary.tmp.parquet"

# Canonical state name aliases (resolves 2018-2025 vs 2026 IMD casing/spelling drift)
STATE_NAME_ALIASES = {
    "ANDAMAN & NICOBAR": "ANDAMAN AND NICOBAR ISLANDS",
    "ANDAMAN & NICOBAR ISLANDS": "ANDAMAN AND NICOBAR ISLANDS",
    "CHHATISGARH": "CHHATTISGARH",
    "DADRA & NAGAR HAVELI": "DADRA AND NAGAR HAVELI",
    "DAMAN & DIU": "DAMAN AND DIU",
    "JAMMU & KASHMIR": "JAMMU AND KASHMIR",
    "LADDAKH": "LADAKH",
    "NCT OF DELHI": "DELHI",
    "ORISSA": "ODISHA",
    "PONDICHERRY": "PUDUCHERRY",
    "TAMILNADU": "TAMIL NADU",
    "UTTARANCHAL": "UTTARAKHAND",
}

SEASON_MONTHS = {
    "Kharif": [6, 7, 8, 9, 10],
    "Rabi": [11, 12, 1, 2, 3],
    "Summer": [3, 4, 5, 6],
    "Autumn": [9, 10, 11],
    "Winter": [12, 1, 2],
    "Whole Year": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
}


def normalize_state_name(state: str) -> str:
    """Normalize state names consistently across IMD sources."""
    cleaned = " ".join(str(state).strip().upper().split())
    return STATE_NAME_ALIASES.get(cleaned, cleaned)

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


def build_district_season_summary() -> pd.DataFrame:
    """Aggregate normalized district-season rainfall summary from integrated district Parquet.

    Normalizes State and District names so 2018-2025 and 2026 IMD sources merge
    into a single canonical row per (State_Clean, District_Upper, Season).
    Never fabricates missing rainfall observations.
    """
    if not DISTRICT_OUTPUT.exists():
        raise FileNotFoundError(f"Missing integrated district rainfall parquet: {DISTRICT_OUTPUT}")

    df = pd.read_parquet(
        DISTRICT_OUTPUT,
        columns=["State", "District", "Year", "Month", "Rainfall"],
    )
    df = df.dropna(subset=["State", "District", "Year", "Month", "Rainfall"]).copy()
    df["State_Clean"] = df["State"].map(normalize_state_name)
    df["District_Upper"] = df["District"].astype(str).str.strip().str.upper()

    season_frames = []
    for season_name, months in SEASON_MONTHS.items():
        sub = df[df["Month"].isin(months)]
        if sub.empty:
            continue

        # Sum daily rainfall per (State, District, Year) for the season
        yearly = (
            sub.groupby(["State_Clean", "District_Upper", "Year"], as_index=False)["Rainfall"]
            .sum()
            .rename(columns={"Rainfall": "seasonal_rainfall_mm"})
        )

        # Aggregate across observed years
        agg = (
            yearly.groupby(["State_Clean", "District_Upper"], as_index=False)["seasonal_rainfall_mm"]
            .agg(
                mean_rainfall_mm="mean",
                min_rainfall_mm="min",
                max_rainfall_mm="max",
                std_rainfall_mm="std",
                years_observed="count",
            )
        )
        agg["Season"] = season_name
        agg["rainfall_cv"] = np.where(
            (agg["years_observed"] >= 2) & (agg["mean_rainfall_mm"] > 0),
            (agg["std_rainfall_mm"] / agg["mean_rainfall_mm"]) * 100.0,
            np.nan,
        )
        for col in ["mean_rainfall_mm", "min_rainfall_mm", "max_rainfall_mm", "std_rainfall_mm", "rainfall_cv"]:
            agg[col] = agg[col].round(2)

        season_frames.append(agg)

    summary_df = pd.concat(season_frames, ignore_index=True)
    summary_df = summary_df[
        [
            "State_Clean",
            "District_Upper",
            "mean_rainfall_mm",
            "min_rainfall_mm",
            "max_rainfall_mm",
            "std_rainfall_mm",
            "years_observed",
            "Season",
            "rainfall_cv",
        ]
    ].sort_values(["State_Clean", "District_Upper", "Season"]).reset_index(drop=True)

    if SEASON_SUMMARY_TEMP.exists():
        SEASON_SUMMARY_TEMP.unlink()
    summary_df.to_parquet(SEASON_SUMMARY_TEMP, index=False, compression="snappy")
    os.replace(SEASON_SUMMARY_TEMP, SEASON_SUMMARY_OUTPUT)
    print(f"[OK] District-season summary saved: {SEASON_SUMMARY_OUTPUT} ({len(summary_df):,} rows)")
    return summary_df


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    print("=" * 85)
    print("RAINFALL DATA INTEGRATION PIPELINE — STAGE 1")
    print("Machine Learning Based Crop Yield Prediction & Recommendation System")
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

