
import pandas as pd

production = pd.read_csv(
    "data/India Agriculture Crop Production.csv"
)

integrated = pd.read_csv(
    "data/production_cost_integrated.csv"
)

print("\n===== INTEGRATION VERIFICATION =====")

# 1. Record count
expected = (
    production["Production Units"]
    .astype("string")
    .str.strip()
    .str.casefold()
    .eq("tonnes")
    .sum()
)

print("Expected production records:", expected)
print("Integrated records:", len(integrated))

# 2. Check production value totals
print("\nProduction total (tonnes):")
print("Original:", production.loc[
    production["Production Units"].astype("string").str.strip().str.casefold() == "tonnes",
    "Production"
].sum())

print("Integrated:", integrated["Production"].sum())

# 3. Check cost status
print("\nCost status:")
print(integrated["cost_status"].value_counts(dropna=False))

# 4. Check missing cost values
print("\nMissing C2 cultivation costs:",
      integrated["cul_cost_c2"].isna().sum())

print("Missing C2 production costs:",
      integrated["prod_cost_c2"].isna().sum())

# 5. Check duplicate production rows
duplicate_count = integrated.duplicated(
    subset=["State", "District", "Crop", "Year", "Season"],
    keep=False
).sum()

print("\nPotential duplicate production rows:", duplicate_count)

print("\n===== VERIFICATION COMPLETED =====")
