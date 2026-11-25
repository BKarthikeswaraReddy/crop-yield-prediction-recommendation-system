"""
Temperature GRD Data Integration Pipeline
Machine Learning Based Crop Yield Prediction & Recommendation System

Decodes IMD binary GRD files for maximum and minimum temperature (2018-2025).
Grid: 31x31 (Lat 7.5-37.5°N, Lon 67.5-97.5°E, 1° resolution).
Each grid cell = 4-byte little-endian float. 99.9 = missing.

Outputs:
    data/weather/temperature_gridwise_integrated.parquet
    data/weather/temperature_monthly_summary.parquet
    data/weather/temperature_integration_summary.csv
"""

import os
import struct
import calendar
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# -------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
WEATHER_DIR = DATA_DIR / "weather"
MAX_TEMP_DIR = WEATHER_DIR / "maximum_temperature"
MIN_TEMP_DIR = WEATHER_DIR / "minmum_temperature"  # Note: original typo preserved

OUTPUT_DAILY = WEATHER_DIR / "temperature_gridwise_integrated.parquet"
OUTPUT_MONTHLY = WEATHER_DIR / "temperature_monthly_summary.parquet"
OUTPUT_SUMMARY = WEATHER_DIR / "temperature_integration_summary.csv"
TEMP_DAILY = WEATHER_DIR / "temperature_gridwise_integrated.tmp.parquet"
TEMP_MONTHLY = WEATHER_DIR / "temperature_monthly_summary.tmp.parquet"

# Grid parameters
NLAT = 31
NLON = 31
LAT_START = 7.5
LON_START = 67.5
MISSING_VALUE = 99.9
MISSING_TOLERANCE = 0.05  # values within 99.9 ± 0.05 treated as missing

# Schemas
DAILY_SCHEMA = pa.schema([
    ("Date", pa.date32()),
    ("Year", pa.int32()),
    ("Month", pa.int32()),
    ("Day", pa.int32()),
    ("Latitude", pa.float64()),
    ("Longitude", pa.float64()),
    ("Max_Temp_C", pa.float64()),
    ("Min_Temp_C", pa.float64()),
    ("Temp_Range_C", pa.float64()),
    ("Max_Temp_Flag", pa.string()),  # valid / missing / anomaly
    ("Min_Temp_Flag", pa.string()),
    ("Source_max", pa.string()),
    ("Source_min", pa.string()),
])

MONTHLY_SCHEMA = pa.schema([
    ("Year", pa.int32()),
    ("Month", pa.int32()),
    ("Latitude", pa.float64()),
    ("Longitude", pa.float64()),
    ("Avg_Max_Temp_C", pa.float64()),
    ("Avg_Min_Temp_C", pa.float64()),
    ("Avg_Temp_Range_C", pa.float64()),
    ("Max_of_Max_Temp_C", pa.float64()),
    ("Min_of_Min_Temp_C", pa.float64()),
    ("Valid_Days", pa.int32()),
    ("Total_Days", pa.int32()),
])


def decode_grd_file(filepath: Path) -> np.ndarray:
    """Decode a binary GRD file into a (days, NLAT, NLON) array.

    Returns float64 array with NaN for missing values (99.9).
    """
    file_size = filepath.stat().st_size
    grid_size = NLAT * NLON
    bytes_per_grid = grid_size * 4
    num_days = file_size // bytes_per_grid

    if file_size % bytes_per_grid != 0:
        raise ValueError(
            f"{filepath.name}: size {file_size} is not divisible by "
            f"{bytes_per_grid} (31x31x4). Cannot decode."
        )

    data = np.fromfile(str(filepath), dtype="<f4")  # little-endian float32
    data = data.astype(np.float64)
    data = data.reshape(num_days, NLAT, NLON)

    # Mark 99.9 as NaN
    missing_mask = np.abs(data - MISSING_VALUE) < MISSING_TOLERANCE
    data[missing_mask] = np.nan

    return data


def process_year(year: int) -> tuple:
    """Process one year of max and min temperature GRD files.

    Returns (daily_df, summary_dict) or (None, summary_dict) if files are invalid.
    """
    max_file = MAX_TEMP_DIR / f"Maxtemp_MaxT_{year}.GRD"
    min_file = MIN_TEMP_DIR / f"Mintemp_MinT_{year}.GRD"

    summary = {
        "year": year,
        "max_file": max_file.name if max_file.exists() else "MISSING",
        "min_file": min_file.name if min_file.exists() else "MISSING",
        "max_file_bytes": max_file.stat().st_size if max_file.exists() else 0,
        "min_file_bytes": min_file.stat().st_size if min_file.exists() else 0,
    }

    if not max_file.exists() or not min_file.exists():
        print(f"  [SKIP] {year}: Missing GRD file(s)")
        summary.update({"status": "missing_files", "output_rows": 0})
        return None, summary

    if max_file.stat().st_size == 0 or min_file.stat().st_size == 0:
        print(f"  [SKIP] {year}: Empty GRD file(s)")
        summary.update({"status": "empty_files", "output_rows": 0})
        return None, summary

    # Decode
    max_data = decode_grd_file(max_file)
    min_data = decode_grd_file(min_file)

    num_days_in_year = 366 if calendar.isleap(year) else 365

    if max_data.shape[0] != num_days_in_year:
        print(f"  [WARN] {year}: Max temp has {max_data.shape[0]} days, expected {num_days_in_year}")
    if min_data.shape[0] != num_days_in_year:
        print(f"  [WARN] {year}: Min temp has {min_data.shape[0]} days, expected {num_days_in_year}")

    # Use the minimum of available days
    num_days = min(max_data.shape[0], min_data.shape[0], num_days_in_year)

    # Build coordinate grids
    lats = np.array([LAT_START + i for i in range(NLAT)])
    lons = np.array([LON_START + j for j in range(NLON)])

    # Generate dates
    start_date = pd.Timestamp(year, 1, 1)
    dates = pd.date_range(start_date, periods=num_days, freq="D")

    rows = []
    total_missing_max = 0
    total_missing_min = 0
    total_anomalies = 0

    for day_idx in range(num_days):
        date = dates[day_idx]
        for lat_idx in range(NLAT):
            for lon_idx in range(NLON):
                max_val = max_data[day_idx, lat_idx, lon_idx]
                min_val = min_data[day_idx, lat_idx, lon_idx]

                # Flags
                max_flag = "valid"
                min_flag = "valid"

                if np.isnan(max_val):
                    max_flag = "missing"
                    total_missing_max += 1
                elif max_val < -60 or max_val > 60:
                    max_flag = "anomaly"
                    total_anomalies += 1

                if np.isnan(min_val):
                    min_flag = "missing"
                    total_missing_min += 1
                elif min_val < -60 or min_val > 60:
                    min_flag = "anomaly"
                    total_anomalies += 1

                # Temperature range (max - min)
                if max_flag == "valid" and min_flag == "valid":
                    temp_range = max_val - min_val
                    # Check: max should not be below min
                    if max_val < min_val:
                        max_flag = "anomaly"
                        min_flag = "anomaly"
                        total_anomalies += 1
                else:
                    temp_range = np.nan

                rows.append((
                    date.date(),
                    int(date.year), int(date.month), int(date.day),
                    lats[lat_idx], lons[lon_idx],
                    max_val, min_val, temp_range,
                    max_flag, min_flag,
                    max_file.name, min_file.name,
                ))

    df = pd.DataFrame(rows, columns=[
        "Date", "Year", "Month", "Day",
        "Latitude", "Longitude",
        "Max_Temp_C", "Min_Temp_C", "Temp_Range_C",
        "Max_Temp_Flag", "Min_Temp_Flag",
        "Source_max", "Source_min",
    ])

    summary.update({
        "status": "processed",
        "num_days": num_days,
        "output_rows": len(df),
        "missing_max": total_missing_max,
        "missing_min": total_missing_min,
        "anomalies": total_anomalies,
        "valid_records": int(((df["Max_Temp_Flag"] == "valid") & (df["Min_Temp_Flag"] == "valid")).sum()),
    })

    return df, summary


def main():
    print("=" * 85)
    print("TEMPERATURE GRD INTEGRATION PIPELINE")
    print("Machine Learning Based Crop Yield Prediction & Recommendation System")
    print("=" * 85)

    # Discover available years
    years = sorted(set(
        [int(f.stem.split("_")[-1]) for f in MAX_TEMP_DIR.glob("Maxtemp_MaxT_*.GRD") if f.stat().st_size > 0]
    ))
    print(f"\nFound temperature data for years: {years}")
    print(f"Max temp dir: {MAX_TEMP_DIR}")
    print(f"Min temp dir: {MIN_TEMP_DIR}")

    # Clean temp files
    for tmp in [TEMP_DAILY, TEMP_MONTHLY]:
        if tmp.exists():
            tmp.unlink()

    summaries: List[Dict] = []
    writer = None

    try:
        writer = pq.ParquetWriter(TEMP_DAILY, DAILY_SCHEMA, compression="snappy")

        for year in years:
            print(f"\nProcessing {year}...")
            df, summary = process_year(year)
            summaries.append(summary)

            if df is not None and len(df) > 0:
                table = pa.Table.from_pandas(df, schema=DAILY_SCHEMA, preserve_index=False)
                writer.write_table(table)
                print(
                    f"  Rows={len(df):,} | Valid={summary['valid_records']:,} | "
                    f"MissingMax={summary['missing_max']:,} | MissingMin={summary['missing_min']:,} | "
                    f"Anomalies={summary['anomalies']}"
                )
                del df, table

        writer.close()
        writer = None
        os.replace(TEMP_DAILY, OUTPUT_DAILY)
        print(f"\n[OK] Daily temperature saved: {OUTPUT_DAILY}")

    except Exception as exc:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if TEMP_DAILY.exists():
            TEMP_DAILY.unlink()
        raise exc

    # ---- Monthly summary ----
    print("\n" + "=" * 85)
    print("COMPUTING MONTHLY TEMPERATURE SUMMARIES")
    print("=" * 85)

    dt = pq.read_table(OUTPUT_DAILY)
    ddf = dt.to_pandas()

    # Only use valid records for aggregation
    valid_mask = (ddf["Max_Temp_Flag"] == "valid") & (ddf["Min_Temp_Flag"] == "valid")

    monthly = ddf[valid_mask].groupby(["Year", "Month", "Latitude", "Longitude"]).agg(
        Avg_Max_Temp_C=("Max_Temp_C", "mean"),
        Avg_Min_Temp_C=("Min_Temp_C", "mean"),
        Avg_Temp_Range_C=("Temp_Range_C", "mean"),
        Max_of_Max_Temp_C=("Max_Temp_C", "max"),
        Min_of_Min_Temp_C=("Min_Temp_C", "min"),
        Valid_Days=("Max_Temp_C", "count"),
    ).reset_index()

    # Total days in each month for each grid
    total_days = ddf.groupby(["Year", "Month", "Latitude", "Longitude"]).size().reset_index(name="Total_Days")
    monthly = monthly.merge(total_days, on=["Year", "Month", "Latitude", "Longitude"], how="left")

    # Cast types
    monthly["Year"] = monthly["Year"].astype("int32")
    monthly["Month"] = monthly["Month"].astype("int32")
    monthly["Valid_Days"] = monthly["Valid_Days"].astype("int32")
    monthly["Total_Days"] = monthly["Total_Days"].astype("int32")

    table = pa.Table.from_pandas(monthly, schema=MONTHLY_SCHEMA, preserve_index=False)
    pq.write_table(table, TEMP_MONTHLY, compression="snappy")
    os.replace(TEMP_MONTHLY, OUTPUT_MONTHLY)
    print(f"Monthly summary rows: {len(monthly):,}")
    print(f"[OK] Monthly temperature saved: {OUTPUT_MONTHLY}")

    del dt, ddf, monthly, total_days, table

    # ---- Summary CSV ----
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)
    print(f"\n[OK] Summary saved: {OUTPUT_SUMMARY}")

    # ---- Verification ----
    print("\n" + "=" * 85)
    print("VERIFICATION")
    print("=" * 85)

    dt = pq.read_table(OUTPUT_DAILY)
    ddf = dt.to_pandas()
    print(f"Daily Parquet: {len(ddf):,} rows, {len(dt.schema)} columns")
    print(f"Schema: {[f.name for f in dt.schema]}")
    print(f"Date range: {ddf['Date'].min()} to {ddf['Date'].max()}")
    print(f"Years: {sorted(ddf['Year'].unique().tolist())}")
    print(f"Grid: {ddf['Latitude'].nunique()} lats x {ddf['Longitude'].nunique()} lons")
    print(f"Max temp range: {ddf['Max_Temp_C'].min():.1f} to {ddf['Max_Temp_C'].max():.1f} °C")
    print(f"Min temp range: {ddf['Min_Temp_C'].min():.1f} to {ddf['Min_Temp_C'].max():.1f} °C")
    print(f"Max_Temp_Flag: {ddf['Max_Temp_Flag'].value_counts().to_dict()}")
    print(f"Min_Temp_Flag: {ddf['Min_Temp_Flag'].value_counts().to_dict()}")

    mt = pq.read_table(OUTPUT_MONTHLY)
    mdf = mt.to_pandas()
    print(f"\nMonthly Parquet: {len(mdf):,} rows")

    print("\n" + "=" * 85)
    print("TEMPERATURE INTEGRATION COMPLETED")
    total_rows = sum(s.get("output_rows", 0) for s in summaries)
    total_missing = sum(s.get("missing_max", 0) + s.get("missing_min", 0) for s in summaries)
    print(f"Total daily grid records: {total_rows:,}")
    print(f"Total missing values: {total_missing:,}")
    print("=" * 85)


if __name__ == "__main__":
    main()
