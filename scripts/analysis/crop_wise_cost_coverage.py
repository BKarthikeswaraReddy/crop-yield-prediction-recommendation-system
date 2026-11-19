
import pandas as pd

production = pd.read_csv("data/India Agriculture Crop Production.csv")
cost = pd.read_csv("data/Cost_of_Cultivation/cost-of-cultivation.csv")

def normalize(series):
    return series.astype("string").str.strip().str.casefold()

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

cost["state_key"] = normalize(cost["state_name"])
cost["crop_key"] = normalize(cost["crop_name"])
cost["year_key"] = normalize(cost["year"])

# Remove rows with missing matching keys
production = production.dropna(
    subset=["state_key", "crop_key", "year_key"]
)

cost = cost.dropna(
    subset=["state_key", "crop_key", "year_key"]
)

production_keys = set(
    zip(production["state_key"], production["crop_key"], production["year_key"])
)

cost_keys = set(
    zip(cost["state_key"], cost["crop_key"], cost["year_key"])
)

rows = []

for crop in sorted(production["crop_key"].unique()):

    p_keys = {
        key for key in production_keys if key[1] == crop
    }

    c_keys = {
        key for key in cost_keys if key[1] == crop
    }

    matched_keys = p_keys & c_keys

    rows.append({
        "Crop": crop,
        "Production combinations": len(p_keys),
        "Cost combinations": len(c_keys),
        "Matching combinations": len(matched_keys),
        "Unmatched production combinations": len(p_keys - c_keys),
        "Coverage percentage": round(
            len(matched_keys) / len(p_keys) * 100, 2
        ) if p_keys else 0
    })

report = pd.DataFrame(rows)

report = report.sort_values(
    by="Coverage percentage",
    ascending=False
)

print("=" * 85)
print("CROP-WISE COST DATA COVERAGE")
print("=" * 85)

print(report.to_string(index=False))

print("\nCrops with zero matching combinations:")
print(report.loc[report["Matching combinations"] == 0, "Crop"].tolist())

print("\nCrops with partial matching:")
print(
    report.loc[
        (report["Matching combinations"] > 0)
        & (report["Unmatched production combinations"] > 0),
        "Crop"
    ].tolist()
)

print("\nCrops with complete matching:")
print(
    report.loc[
        report["Coverage percentage"] == 100,
        "Crop"
    ].tolist()
)

report.to_csv("crop_wise_cost_coverage.csv", index=False)

print("\nReport saved: crop_wise_cost_coverage.csv")
print("=" * 85)
