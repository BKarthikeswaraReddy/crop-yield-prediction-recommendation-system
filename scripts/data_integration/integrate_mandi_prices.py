"""
Mandi Prices Integration Pipeline
Machine Learning Based Crop Yield Prediction & Recommendation System

Consolidates 26 yearly mandi price Parquet files (2001-2026) into one memory-efficient,
schema-enforced Parquet dataset using PyArrow ParquetWriter streaming.

Key Principles:
- Single-year processing for minimal memory footprint.
- Exact normalized matching against crop_name_mapping.csv.
- Full record preservation: records with invalid dates, missing/non-positive modal prices,
  or duplicate transaction keys are retained with explicit quality flags, never silently dropped.
- Preserves all 11 original mandi columns and adds 11 standardized metadata and audit fields.
- Missing numerical values remain null (never replaced with zero).
- Strict separation of provisional 'candidate' and 'review' mappings.
- Safe output handling: writes to a temporary Parquet file first, atomically replacing the final
  output only upon 100% successful completion across all 26 years.
- Comprehensive year-by-year accounting audit saved to CSV, including source-level quality metrics
  and semicolon-separated unmatched commodity names.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# -----------------------------------------------------------------------------
# 1. PATH CONFIGURATION
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
MANDI_DIR = DATA_DIR / "mandi_prices"
MAPPING_FILE = SCRIPT_DIR / "crop_name_mapping.csv"

# Safe temporary and final output paths
OUTPUT_FILE = DATA_DIR / "mandi_prices_integrated.parquet"
TEMP_OUTPUT_FILE = DATA_DIR / "mandi_prices_integrated.tmp.parquet"
SUMMARY_FILE = DATA_DIR / "mandi_integration_summary.csv"

# -----------------------------------------------------------------------------
# 2. TARGET PYARROW SCHEMA (22 COLUMNS)
# -----------------------------------------------------------------------------
# Preserves all 11 original mandi columns (1-11) + 11 metadata/audit columns (12-22)
# All types are nullable to properly represent real-world missingness without sentinel values.

TARGET_SCHEMA = pa.schema([
    # Original 11 mandi columns
    ("State", pa.string()),
    ("District", pa.string()),
    ("Market", pa.string()),
    ("Commodity", pa.string()),
    ("Variety", pa.string()),
    ("Grade", pa.string()),
    ("Arrival_Date", pa.date32()),
    ("Min_Price", pa.float64()),
    ("Max_Price", pa.float64()),
    ("Modal_Price", pa.float64()),
    ("Commodity_Code", pa.int64()),

    # Added mapping and classification fields
    ("production_crop", pa.string()),
    ("mandi_commodity", pa.string()),
    ("match_type", pa.string()),
    ("status", pa.string()),
    ("is_candidate_mapping", pa.bool_()),
    ("is_review_mapping", pa.bool_()),
    ("source_year", pa.int32()),

    # Quality and audit flags
    ("is_valid_date", pa.bool_()),
    ("is_valid_modal_price", pa.bool_()),
    ("is_duplicate", pa.bool_()),

    # Raw date preservation
    ("Arrival_Date_raw", pa.string()),
])

REQUIRED_MANDI_COLUMNS = [
    "State",
    "District",
    "Market",
    "Commodity",
    "Variety",
    "Grade",
    "Arrival_Date",
    "Min_Price",
    "Max_Price",
    "Modal_Price",
    "Commodity_Code"
]

OUTPUT_COLUMNS = [field.name for field in TARGET_SCHEMA]


# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS & VALIDATION
# -----------------------------------------------------------------------------

def load_and_validate_mapping(mapping_path: Path) -> pd.DataFrame:
    """Load and validate the crop mapping CSV.

    Ensures no duplicate commodity keys exist to prevent cartesian product joins.
    """
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping CSV file not found at: {mapping_path}")

    mapping = pd.read_csv(mapping_path)

    required_cols = {"production_crop", "mandi_commodity", "match_type", "status"}
    missing_cols = required_cols - set(mapping.columns)
    if missing_cols:
        raise ValueError(f"Mapping CSV missing required columns: {missing_cols}")

    mapping = mapping.dropna(subset=["production_crop", "mandi_commodity", "status"]).copy()

    # Create normalized matching key: stripped and case-folded
    mapping["commodity_key"] = (
        mapping["mandi_commodity"].astype(str).str.strip().str.casefold()
    )

    # Disallow ambiguous mappings (same commodity key pointing to multiple crops)
    if mapping["commodity_key"].duplicated().any():
        duplicated_items = mapping.loc[
            mapping["commodity_key"].duplicated(keep=False), "mandi_commodity"
        ].tolist()
        raise ValueError(
            f"Ambiguous mapping detected! Commodities mapped multiple times: {duplicated_items}"
        )

    mapping["production_crop"] = mapping["production_crop"].astype(str).str.strip()
    mapping["match_type"] = mapping["match_type"].astype(str).str.strip()
    mapping["status"] = mapping["status"].astype(str).str.strip().str.lower()

    # Validate mapping statuses
    valid_statuses = {"candidate", "review"}
    found_statuses = set(mapping["status"].unique())
    invalid_statuses = found_statuses - valid_statuses
    if invalid_statuses:
        raise ValueError(f"Unexpected mapping status(es) found: {invalid_statuses}")

    return mapping


def extract_year_from_filename(filename: str) -> int:
    """Extract 4-digit year from filename such as mandi_daily_2001.parquet."""
    match = re.search(r"(\d{4})", filename)
    if not match:
        raise ValueError(f"Cannot extract 4-digit year from filename: {filename}")
    return int(match.group(1))


# -----------------------------------------------------------------------------
# 4. MAIN INTEGRATION ENGINE
# -----------------------------------------------------------------------------

def run_integration():
    """Execute the full 26-year mandi prices integration pipeline."""
    print("=" * 85)
    print("MANDI PRICES INTEGRATION PIPELINE")
    print("Machine Learning Based Crop Yield Prediction & Recommendation System")
    print("=" * 85)

    # 1. Load and validate crop mapping CSV
    print(f"\n[1/4] Loading crop name mapping: {MAPPING_FILE.name}")
    mapping = load_and_validate_mapping(MAPPING_FILE)
    total_mappings = len(mapping)
    candidate_crops = int((mapping["status"] == "candidate").sum())
    review_crops = int((mapping["status"] == "review").sum())

    print(f"      Total Mappings Loaded: {total_mappings}")
    print(f"      - Candidate mappings (provisional):   {candidate_crops}")
    print(f"      - Review mappings (requires verify): {review_crops}")

    mapping_lookup = mapping.set_index("commodity_key")[[
        "production_crop", "mandi_commodity", "match_type", "status"
    ]]

    # 2. Discover and validate all 26 yearly Parquet files (2001 to 2026)
    print(f"\n[2/4] Verifying yearly source Parquet files in: {MANDI_DIR.name}")
    yearly_files = sorted(MANDI_DIR.glob("mandi_daily_*.parquet"))

    if len(yearly_files) != 26:
        raise ValueError(
            f"Expected exactly 26 yearly Parquet files (2001-2026), found {len(yearly_files)} in {MANDI_DIR}"
        )

    # Verify that each year from 2001 to 2026 is present exactly once
    found_years = [extract_year_from_filename(f.name) for f in yearly_files]
    expected_years = list(range(2001, 2027))
    if found_years != expected_years:
        missing_years = set(expected_years) - set(found_years)
        raise ValueError(f"Missing or duplicate years in input files. Missing: {missing_years}")

    print(f"      Verified all 26 years (2001 to 2026) are present exactly once.")
    print(f"      Target Output:    {OUTPUT_FILE}")
    print(f"      Temporary Output: {TEMP_OUTPUT_FILE}")
    print(f"      Summary Report:   {SUMMARY_FILE}\n")

    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Remove any stale temporary file from an interrupted prior run
    if TEMP_OUTPUT_FILE.exists():
        TEMP_OUTPUT_FILE.unlink()

    # Global accounting collectors
    summary_records: List[Dict] = []
    unmatched_commodities_global: Set[str] = set()

    total_source_records = 0
    total_matched_records = 0
    total_unmatched_records = 0
    total_candidate_records = 0
    total_review_records = 0

    # Source-level quality audit collectors across ALL records
    total_source_invalid_dates = 0
    total_source_invalid_prices = 0

    # Matched-record quality audit collectors
    total_matched_invalid_dates = 0
    total_matched_invalid_prices = 0
    total_duplicates = 0
    total_valid_integrated = 0

    writer: pq.ParquetWriter = None

    try:
        # Initialize safe PyArrow ParquetWriter streaming to the temporary file
        writer = pq.ParquetWriter(
            TEMP_OUTPUT_FILE,
            TARGET_SCHEMA,
            compression="snappy"
        )

        print("[3/4] Processing yearly files sequentially (Memory-Safe Streaming)...")

        for idx, file_path in enumerate(yearly_files, 1):
            year = extract_year_from_filename(file_path.name)
            df = pd.read_parquet(file_path)

            # Validate input columns
            missing_mandi_cols = set(REQUIRED_MANDI_COLUMNS) - set(df.columns)
            if missing_mandi_cols:
                raise ValueError(
                    f"{file_path.name} is missing required columns: {missing_mandi_cols}"
                )

            source_records = len(df)
            total_source_records += source_records

            # -------------------------------------------------------------
            # CORRECTION 2: DATA QUALITY AUDIT ACROSS ALL SOURCE RECORDS
            # Calculated across all source records BEFORE commodity matching
            # -------------------------------------------------------------
            source_parsed_dates = pd.to_datetime(df["Arrival_Date"], errors="coerce")
            source_invalid_dates = int(source_parsed_dates.isna().sum())
            total_source_invalid_dates += source_invalid_dates

            source_numeric_modal = pd.to_numeric(df["Modal_Price"], errors="coerce")
            source_invalid_prices = int(
                (source_numeric_modal.isna() | (source_numeric_modal <= 0)).sum()
            )
            total_source_invalid_prices += source_invalid_prices

            # Create normalized matching key: strip whitespace and case-fold
            df["commodity_key"] = (
                df["Commodity"].astype(str).str.strip().str.casefold()
            )

            # Match against mapping table
            matched_mask = df["commodity_key"].isin(mapping_lookup.index)
            unmatched_df = df.loc[~matched_mask]
            matched_df = df.loc[matched_mask].copy()

            matched_records = len(matched_df)
            unmatched_records = len(unmatched_df)

            # Strict accounting check: source_records = matched_records + unmatched_records
            if source_records != matched_records + unmatched_records:
                raise ValueError(
                    f"Accounting violation in {file_path.name}: "
                    f"source ({source_records}) != matched ({matched_records}) + unmatched ({unmatched_records})"
                )

            total_matched_records += matched_records
            total_unmatched_records += unmatched_records

            # -------------------------------------------------------------
            # CORRECTION 3: UNMATCHED COMMODITY REPORTING
            # Save sorted unique unmatched commodity names as a semicolon list
            # -------------------------------------------------------------
            year_unmatched_commodities = sorted(
                unmatched_df["Commodity"].astype(str).str.strip().unique().tolist()
            )
            unmatched_commodity_names_str = "; ".join(year_unmatched_commodities)
            unmatched_commodities_global.update(year_unmatched_commodities)

            # Handle edge case: no records matched for a year
            if matched_records == 0:
                print(f"[{idx:02d}/26] Year {year}: 0 matched records out of {source_records:,} source records.")
                summary_records.append({
                    "year": year,
                    "source_file": file_path.name,
                    "source_records": source_records,
                    "matched_records": 0,
                    "unmatched_records": unmatched_records,
                    "unmatched_unique_commodities": len(year_unmatched_commodities),
                    "unmatched_commodity_names": unmatched_commodity_names_str,
                    "candidate_records": 0,
                    "review_records": 0,
                    "source_invalid_date_records": source_invalid_dates,
                    "source_invalid_price_records": source_invalid_prices,
                    "invalid_date_records": 0,
                    "invalid_price_records": 0,
                    "records_with_invalid_date_or_price": 0,
                    "duplicate_records": 0,
                    "valid_integrated_records": 0,
                })
                del df, unmatched_df, matched_df, source_parsed_dates, source_numeric_modal
                continue

            # Merge mapping attributes into matched records (inner/left join with unique keys)
            merged = matched_df.merge(
                mapping_lookup,
                left_on="commodity_key",
                right_index=True,
                how="left"
            )

            # -------------------------------------------------------------
            # METADATA & STATUS CLASSIFICATION
            # -------------------------------------------------------------
            merged["is_candidate_mapping"] = (merged["status"] == "candidate").astype(bool)
            merged["is_review_mapping"] = (merged["status"] == "review").astype(bool)
            merged["source_year"] = int(year)

            candidate_records = int(merged["is_candidate_mapping"].sum())
            review_records = int(merged["is_review_mapping"].sum())

            # Accounting check: matched_records = candidate_records + review_records
            if matched_records != candidate_records + review_records:
                raise ValueError(
                    f"Mapping status accounting violation in {file_path.name}: "
                    f"matched ({matched_records}) != candidate ({candidate_records}) + review ({review_records})"
                )

            total_candidate_records += candidate_records
            total_review_records += review_records

            # -------------------------------------------------------------
            # DATA QUALITY AUDIT & RECORD PRESERVATION (MATCHED RECORDS)
            # -------------------------------------------------------------
            # Preserve original date as string before any parsing to prevent data loss
            merged["Arrival_Date_raw"] = merged["Arrival_Date"].astype(str)

            # 1. Date parsing and validation for matched records
            parsed_dates = pd.to_datetime(merged["Arrival_Date"], errors="coerce")
            is_valid_date_series = parsed_dates.notna()
            merged["is_valid_date"] = is_valid_date_series.astype(bool)

            # Store parsed date (date32); invalid dates become null (None)
            valid_dates = parsed_dates.dt.date
            merged["Arrival_Date"] = valid_dates.where(is_valid_date_series, None)

            invalid_date_records = int((~is_valid_date_series).sum())
            total_matched_invalid_dates += invalid_date_records

            # 2. Modal price validation for matched records
            numeric_modal_prices = pd.to_numeric(merged["Modal_Price"], errors="coerce")
            is_valid_modal_price_series = numeric_modal_prices.notna() & (numeric_modal_prices > 0)
            merged["is_valid_modal_price"] = is_valid_modal_price_series.astype(bool)

            # Preserve numerical values as nullable floats (never replace missing values with zero)
            merged["Modal_Price"] = numeric_modal_prices.astype(float)
            merged["Min_Price"] = pd.to_numeric(merged["Min_Price"], errors="coerce").astype(float)
            merged["Max_Price"] = pd.to_numeric(merged["Max_Price"], errors="coerce").astype(float)

            invalid_price_records = int((~is_valid_modal_price_series).sum())
            total_matched_invalid_prices += invalid_price_records

            # 3. Commodity code: preserve nulls as nullable integer
            merged["Commodity_Code"] = pd.to_numeric(
                merged["Commodity_Code"], errors="coerce"
            ).astype("Int64")

            # -------------------------------------------------------------
            # CORRECTION 1: DUPLICATE DETECTION WITH keep=False
            # Every row belonging to a duplicate group is flagged; all rows kept
            # -------------------------------------------------------------
            key_transaction_cols = [
                "State",
                "District",
                "Market",
                "Commodity",
                "Variety",
                "Grade",
                "Arrival_Date_raw"
            ]
            merged["is_duplicate"] = merged.duplicated(
                subset=key_transaction_cols,
                keep=False
            ).astype(bool)

            duplicate_records = int(merged["is_duplicate"].sum())
            total_duplicates += duplicate_records

            # 5. Combined validity: records with BOTH valid date and valid modal price
            is_clean_record_series = is_valid_date_series & is_valid_modal_price_series
            valid_integrated_records = int(is_clean_record_series.sum())
            records_with_invalid_date_or_price = int((~is_clean_record_series).sum())

            # Accounting check without double counting:
            # matched_records = valid_integrated_records + records_with_invalid_date_or_price
            if matched_records != valid_integrated_records + records_with_invalid_date_or_price:
                raise ValueError(
                    f"Validity accounting violation in {file_path.name}: "
                    f"matched ({matched_records}) != valid ({valid_integrated_records}) + invalid ({records_with_invalid_date_or_price})"
                )

            total_valid_integrated += valid_integrated_records

            # Ensure all string columns are clean stripped strings
            string_cols = [
                "State", "District", "Market", "Commodity", "Variety", "Grade",
                "production_crop", "mandi_commodity", "match_type", "status",
                "Arrival_Date_raw"
            ]
            for col in string_cols:
                merged[col] = merged[col].astype(str).str.strip()

            # Align DataFrame exactly with TARGET_SCHEMA (all matched rows preserved)
            export_df = merged[OUTPUT_COLUMNS]

            # -------------------------------------------------------------
            # STREAM DIRECTLY TO TEMPORARY PARQUET VIA PARQUETWRITER
            # -------------------------------------------------------------
            table = pa.Table.from_pandas(
                export_df,
                schema=TARGET_SCHEMA,
                preserve_index=False
            )
            writer.write_table(table)

            # Record year-wise summary row with source-level audit and unmatched names
            summary_records.append({
                "year": year,
                "source_file": file_path.name,
                "source_records": source_records,
                "matched_records": matched_records,
                "unmatched_records": unmatched_records,
                "unmatched_unique_commodities": len(year_unmatched_commodities),
                "unmatched_commodity_names": unmatched_commodity_names_str,
                "candidate_records": candidate_records,
                "review_records": review_records,
                "source_invalid_date_records": source_invalid_dates,
                "source_invalid_price_records": source_invalid_prices,
                "invalid_date_records": invalid_date_records,
                "invalid_price_records": invalid_price_records,
                "records_with_invalid_date_or_price": records_with_invalid_date_or_price,
                "duplicate_records": duplicate_records,
                "valid_integrated_records": valid_integrated_records,
            })

            # Print annual progress
            print(
                f"[{idx:02d}/26] Year {year}: Source={source_records:,} | "
                f"Matched={matched_records:,} (Candidate={candidate_records:,}, Review={review_records:,}) | "
                f"Unmatched={unmatched_records:,} ({len(year_unmatched_commodities)} crops) | "
                f"SourceAudit: (InvDates={source_invalid_dates}, InvPrices={source_invalid_prices}) | "
                f"MatchedAudit: (Valid={valid_integrated_records:,}, InvDate/Price={records_with_invalid_date_or_price}, DupRows={duplicate_records})"
            )

            # Free memory explicitly
            del df, unmatched_df, matched_df, merged, export_df, table, source_parsed_dates, source_numeric_modal

        # Successfully streamed all 26 years! Close writer.
        writer.close()
        writer = None

        # -------------------------------------------------------------
        # ATOMIC PROMOTION OF TEMPORARY FILE TO FINAL OUTPUT
        # -------------------------------------------------------------
        # Only after all 26 years complete without errors, replace final output.
        os.replace(TEMP_OUTPUT_FILE, OUTPUT_FILE)
        print(f"\n[OK] Safely finalized integrated dataset: {OUTPUT_FILE}")

    except Exception as exc:
        print(f"\n[ERROR] Pipeline aborted due to error: {exc}", file=sys.stderr)
        # Close writer if still open
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
            writer = None
        # Clean up temporary file to prevent corruption, leaving previous final output untouched
        if TEMP_OUTPUT_FILE.exists():
            try:
                TEMP_OUTPUT_FILE.unlink()
                print("[INFO] Cleaned up temporary Parquet file.", file=sys.stderr)
            except Exception:
                pass
        raise exc

    # 3. Save Summary CSV Report
    print(f"\n[4/4] Generating integration summary CSV: {SUMMARY_FILE.name}")
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    print(f"      Saved summary report to: {SUMMARY_FILE}")

    # 4. Final Executive Summary
    print("\n" + "=" * 85)
    print("MANDI PRICES INTEGRATION SUMMARY (2001 - 2026)")
    print("=" * 85)
    print(f"Years Processed:                     26 yearly Parquet files (2001 through 2026)")
    print(f"Total Source Records:                {total_source_records:,}")
    print(f"Total Matched Records:               {total_matched_records:,} ({(total_matched_records / max(1, total_source_records) * 100):.2f}%)")
    print(f"Total Unmatched Records:             {total_unmatched_records:,} ({(total_unmatched_records / max(1, total_source_records) * 100):.2f}%)")
    print(f"Distinct Unmapped Commodities:       {len(unmatched_commodities_global):,}")
    print(f"\nClassification of Matched Records:")
    print(f"  - Candidate Mappings (provisional): {total_candidate_records:,}")
    print(f"  - Review Mappings (verification):    {total_review_records:,}")
    print(f"\nData Quality Audit Across ALL Source Records (Pre-Matching):")
    print(f"  - Source Invalid / Missing Dates:    {total_source_invalid_dates:,}")
    print(f"  - Source Missing / Non-Positive:     {total_source_invalid_prices:,}")
    print(f"\nData Quality Audit for Matched Records (Post-Matching):")
    print(f"  - Valid (Clean Date & Price):        {total_valid_integrated:,}")
    print(f"  - Matched Invalid / Missing Dates:   {total_matched_invalid_dates:,}")
    print(f"  - Matched Missing / Invalid Prices:  {total_matched_invalid_prices:,}")
    print(f"  - Duplicate Key Rows (keep=False):   {total_duplicates:,}")
    print(f"\nAccounting Checks:")
    print(f"  - Source = Matched + Unmatched:      {total_source_records == total_matched_records + total_unmatched_records}")
    print(f"  - Matched = Candidate + Review:      {total_matched_records == total_candidate_records + total_review_records}")
    print(f"\nOutput Paths:")
    print(f"  - Integrated Parquet Dataset:        {OUTPUT_FILE}")
    print(f"  - Summary Report CSV:                {SUMMARY_FILE}")
    print("=" * 85)


if __name__ == "__main__":
    run_integration()
