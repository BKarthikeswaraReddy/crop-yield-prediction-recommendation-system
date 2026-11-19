
import pandas as pd

input_file = "data/Cost_of_Cultivation/cost-of-cultivation.csv"

output_file = (
    "data/Cost_of_Cultivation/"
    "cost-of-cultivation-cleaned.csv"
)

df = pd.read_csv(input_file)

# Add data quality flags
df["a2_cost_flag"] = "valid"
df["b1_cost_flag"] = "valid"
df["c3_cost_flag"] = "valid"

# Negative A2 cost: mark unavailable
negative_a2 = df["cul_cost_a2"] < 0

df.loc[negative_a2, "cul_cost_a2"] = pd.NA
df.loc[negative_a2, "a2_cost_flag"] = "negative_value_marked_missing"

# Missing B1 cost: preserve missing value and flag it
missing_b1 = df["prod_cost_b1"].isna()

df.loc[missing_b1, "b1_cost_flag"] = "missing_value"

# Zero C3 cost: mark unavailable pending verification
zero_c3 = df["prod_cost_c3"] == 0

df.loc[zero_c3, "prod_cost_c3"] = pd.NA
df.loc[zero_c3, "c3_cost_flag"] = "zero_value_marked_missing"

# Save a separate cleaned copy
df.to_csv(output_file, index=False)

print("\n===== CLEANING SUMMARY =====")
print("Original records:", len(df))
print("Negative A2 values handled:", int(negative_a2.sum()))
print("Missing B1 values flagged:", int(missing_b1.sum()))
print("Zero C3 values handled:", int(zero_c3.sum()))

print("\nC2 cultivation cost missing:", df["cul_cost_c2"].isna().sum())
print("C2 production cost missing:", df["prod_cost_c2"].isna().sum())

print("\nCleaned dataset saved:")
print(output_file)
