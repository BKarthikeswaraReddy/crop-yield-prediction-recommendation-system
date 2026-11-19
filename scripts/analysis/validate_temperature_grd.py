
from pathlib import Path
import numpy as np

max_folder = Path("data/weather/maximum_temperature")
min_folder = Path("data/weather/minmum_temperature")

MISSING_VALUE = 99.9
GRID_SIZE = 31 * 31

print("=" * 75)
print("IMD TEMPERATURE DATA VALIDATION")
print("=" * 75)

temperature_data = {}

for label, folder, prefix in [
    ("MAXIMUM", max_folder, "Maxtemp_MaxT"),
    ("MINIMUM", min_folder, "Mintemp_MinT")
]:

    print(f"\n{label} TEMPERATURE")
    print("-" * 75)

    for year in range(2018, 2026):

        file_path = folder / f"{prefix}_{year}.GRD"

        if not file_path.exists():
            print(f"\n{year}: FILE NOT FOUND")
            continue

        values = np.fromfile(file_path, dtype="<f4")

        expected_days = 366 if (
            year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        ) else 365

        expected_values = expected_days * GRID_SIZE

        print(f"\nYear: {year}")
        print("Total values:", len(values))
        print("Expected values:", expected_values)
        print("File size valid:", len(values) == expected_values)

        if len(values) != expected_values:
            print("ERROR: Unexpected file dimensions.")
            continue

        grids = values.reshape(expected_days, 31, 31)

        missing = np.isclose(values, MISSING_VALUE)
        valid = values[~missing]

        print("Days:", expected_days)
        print("Grid shape:", grids.shape)
        print("Missing markers (99.9):", int(missing.sum()))
        print("Missing percentage:", round(missing.mean() * 100, 3), "%")
        print("Valid values:", len(valid))

        if len(valid) > 0:
            print("Minimum temperature value:", float(valid.min()))
            print("Maximum temperature value:", float(valid.max()))
            print("Average temperature value:", round(float(valid.mean()), 2))

        print("NaN values:", int(np.isnan(values).sum()))
        print("Infinite values:", int(np.isinf(values).sum()))

        temperature_data[(label, year)] = grids

print("\n" + "=" * 75)
print("MINIMUM VS MAXIMUM TEMPERATURE CHECK")
print("=" * 75)

for year in range(2018, 2026):

    max_grid = temperature_data.get(("MAXIMUM", year))
    min_grid = temperature_data.get(("MINIMUM", year))

    if max_grid is None or min_grid is None:
        continue

    valid = (
        (max_grid != MISSING_VALUE)
        & (min_grid != MISSING_VALUE)
        & np.isfinite(max_grid)
        & np.isfinite(min_grid)
    )

    violations = valid & (min_grid > max_grid)

    print(f"\nYear: {year}")
    print("Comparable observations:", int(valid.sum()))
    print("Minimum greater than maximum:", int(violations.sum()))

print("\nTEMPERATURE VALIDATION COMPLETED")
print("=" * 75)
