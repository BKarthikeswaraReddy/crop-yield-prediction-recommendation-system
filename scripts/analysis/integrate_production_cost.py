
import pandas as pd

# Load datasets
production = pd.read_csv(
    "data/India Agriculture Crop Production.csv"
)

cost = pd.read_csv(
    "data/Cost_of_Cultivation/cost-of-cultivation-cleaned.csv"
)

# Normalize matching keys
def normalize(series):
    return series.astype("string").str.strip().str.casefold()

# Standardize crop names
aliases = {
    "arhar/tur": "tur (arhar)",
    "cotton(lint)": "cotton",
    "masoor": "masur",
    "moong(green gram)": "moong",
    "niger seed": "nigerseed",
    "rapeseed &mustard": "rapeseed & mustard (toria/taramira)",
    "sesamum": "sesamum (til)",
    "rice": "paddy",
}

production["state_key"] = normalize(production["State"])
production["crop_key"] = normalize(production["Crop"]).replace(aliases)
production["year_key"] = normalize(production["Year"])
production["unit_key"] = normalize(production["Production Units"])

cost["state_key"] = normalize(cost["state_name"])
cost["crop_key"] = normalize(cost["crop_name"])
cost["year_key"] = normalize(cost["year"])

# Use tonnes records only
production = production[
    production["unit_key"] == "tonnes"
].copy()

# Keep rows with complete matching keys
valid_production = production.dropna(
    subset=["state_key", "crop_key", "year_key"]
).copy()

# Remove cost rows with incomplete keys
cost = cost.dropna(
    subset=["state_key", "crop_key", "year_key"]
).copy()

keys = ["state_key", "crop_key", "year_key"]

# Identify ambiguous cost combinations
duplicate_mask = cost.duplicated(
    subset=keys,
    keep=False
)

ambiguous_cost = cost[duplicate_mask].copy()
unique_cost = cost[~duplicate_mask].copy()

ambiguous_keys = set(
    zip(
        ambiguous_cost["state_key"],
        ambiguous_cost["crop_key"],
        ambiguous_cost["year_key"]
    )
)

# Select cost fields for integration
cost_columns = [
    "state_key",
    "crop_key",
    "year_key",
    "cul_cost_c2",
    "prod_cost_c2"
]

unique_cost = unique_cost[cost_columns].copy()

# Attach only unambiguous cost records
merged = valid_production.merge(
    unique_cost,
    on=keys,
    how="left",
    validate="many_to_one"
)

# Mark ambiguous combinations separately
merged["cost_status"] = "cost_unavailable"

matched = merged["cul_cost_c2"].notna()

merged.loc[matched, "cost_status"] = "matched"

merged_keys = list(zip(
    merged["state_key"],
    merged["crop_key"],
    merged["year_key"]
))

ambiguous_rows = [
    key in ambiguous_keys for key in merged_keys
]

merged.loc[ambiguous_rows, "cost_status"] = "ambiguous_cost_record"

# Save integrated dataset
output_file = "data/production_cost_integrated.csv"

merged.to_csv(output_file, index=False)

# Summary
print("\n===== PRODUCTION + COST INTEGRATION =====")

print("Production records in tonnes:", len(valid_production))
print("Unique cost records used:", len(unique_cost))
print("Ambiguous cost combinations:", len(ambiguous_keys))

print("\n===== COST MATCHING SUMMARY =====")
print(merged["cost_status"].value_counts().to_string())

print("\nMatching percentage:",
      round((merged["cost_status"] == "matched").mean() * 100, 2),
      "%")

print("\nIntegrated dataset saved:")
print(output_file)

print("\n===== INTEGRATION COMPLETED =====")
