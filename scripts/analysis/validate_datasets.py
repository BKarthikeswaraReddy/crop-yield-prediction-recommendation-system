
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = Path("data")
REPORT_FILE = Path("dataset_validation_report.txt")

report = []

def log(message=""):
    print(message)
    report.append(str(message))

log("=" * 75)
log("SMART AGRICULTURE PROJECT - DATASET INVENTORY")
log("=" * 75)

if not DATA_DIR.exists():
    log("ERROR: data folder not found.")
else:
    for file in sorted(DATA_DIR.rglob("*")):
        if not file.is_file():
            continue

        size_kb = file.stat().st_size / 1024
        log(f"\nFILE: {file}")
        log(f"SIZE: {size_kb:.2f} KB")

        if file.stat().st_size == 0:
            log("WARNING: Empty file - cannot be used.")
            continue

        suffix = file.suffix.lower()

        try:
            if suffix == ".csv":
                df = pd.read_csv(file, nrows=5)
                log("FORMAT: CSV")
                log(f"COLUMNS: {list(df.columns)}")
                log(f"SAMPLE ROWS: {len(df)}")

            elif suffix == ".parquet":
                parquet_file = pq.ParquetFile(file)
                log("FORMAT: PARQUET")
                log(f"ROWS: {parquet_file.metadata.num_rows:,}")
                log(f"COLUMNS: {parquet_file.metadata.num_columns}")
                log(f"SCHEMA: {parquet_file.schema.names}")

            elif suffix in [".xlsx", ".xls"]:
                excel = pd.ExcelFile(file)
                log("FORMAT: EXCEL")
                log(f"SHEETS: {excel.sheet_names}")

            elif suffix == ".grd":
                log("FORMAT: BINARY GRD")
                log("SIZE CHECK: Non-zero file; internal validity needs decoding.")

            else:
                log(f"FORMAT: {suffix}")

        except Exception as error:
            log(f"ERROR READING FILE: {error}")

log("\n" + "=" * 75)
log("INVENTORY COMPLETED")
log("=" * 75)

REPORT_FILE.write_text("\n".join(report), encoding="utf-8")
print(f"\nReport saved to: {REPORT_FILE.resolve()}")
