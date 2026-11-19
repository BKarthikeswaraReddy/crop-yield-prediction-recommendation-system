
import pandas as pd

original = pd.read_csv(
    "data/Cost_of_Cultivation/cost-of-cultivation.csv"
)

cleaned = pd.read_csv(
    "data/Cost_of_Cultivation/cost-of-cultivation-cleaned.csv"
)

print("\n===== CLEANED DATASET VERIFICATION =====")

# 1. Record count
print("\nOriginal records:", len(original))
print("Cleaned records:", len(cleaned))

# 2. Check primary cost columns
for column in ["cul_cost_c2", "prod_cost_c2"]:
    unchanged = original[column].equals(cleaned[column])
    print(f"{column} unchanged:", unchanged)

# 3. Check anomaly flags
print("\nA2 flags:")
print(cleaned["a2_cost_flag"].value_counts())

print("\nB1 flags:")
print(cleaned["b1_cost_flag"].value_counts())

print("\nC3 flags:")
print(cleaned["c3_cost_flag"].value_counts())

# 4. Check duplicate combinations remain preserved
keys = ["state_name", "crop_name", "year"]

duplicates = cleaned[
    cleaned.duplicated(subset=keys, keep=False)
]

print("\nDuplicate records preserved:", len(duplicates))
print("Duplicate combinations:", duplicates[keys].drop_duplicates().shape[0])

# 5. Final checks
print("\nC2 cultivation cost missing:", cleaned["cul_cost_c2"].isna().sum())
print("C2 production cost missing:", cleaned["prod_cost_c2"].isna().sum())

print("\n===== VERIFICATION COMPLETED =====")
