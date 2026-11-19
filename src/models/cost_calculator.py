"""
Cultivation Cost Calculator
Crop Production Prediction and Farmer Decision Support System

Retrieves and calculates C2 Cost of Cultivation (per hectare) and C2 Cost of Production (per quintal)
from official Commission for Agricultural Costs and Prices (CACP) survey data.

Important Principles:
- Respects state, crop, and year coverage.
- Never fabricates costs or missing values.
- Explicitly labels state-level surveys vs national benchmarks vs unobserved data.
- Does not treat state-level cultivation costs as district-specific.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class CultivationCostCalculator:

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            self.data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)

        self.cost_csv_path = self.data_dir / "Cost_of_Cultivation" / "cost-of-cultivation-cleaned.csv"
        self.df_cost = None

        # Normalized mapping from Production crop names to CACP survey crop names
        self.crop_to_cacp = {
            "RICE": "Paddy",
            "PADDY": "Paddy",
            "WHEAT": "Wheat",
            "MAIZE": "Maize",
            "CORN": "Maize",
            "COTTON(LINT)": "Cotton",
            "COTTON": "Cotton",
            "GRAM": "Gram",
            "CHICKPEAS": "Gram",
            "RAPESEED &MUSTARD": "Rapeseed & Mustard (Toria/Taramira)",
            "MUSTARD": "Rapeseed & Mustard (Toria/Taramira)",
            "ARHAR/TUR": "Tur (Arhar)",
            "TUR": "Tur (Arhar)",
            "PULSES": "Gram",
            "URAD": "Urad",
            "SUGARCANE": "Sugarcane",
            "GROUNDNUT": "Groundnut",
            "SESAMUM": "Sesamum (Til)",
            "MOONG(GREEN GRAM)": "Moong",
            "MOONG": "Moong",
            "JOWAR": "Jowar",
            "BAJRA": "Bajra",
            "POTATO": "Potato",
            "SOYABEAN": "Soyabean",
            "SOYBEAN": "Soyabean",
            "ONION": "Onion",
            "JUTE": "Jute",
            "RAGI": "Ragi",
            "MASOOR": "Masur",
            "SUNFLOWER": "Sunflower",
            "BARLEY": "Barley",
            "SAFFLOWER": "Safflower",
            "NIGER SEED": "Nigerseed",
            "LENTIL": "Lentil",
            "PEAS & BEANS (PULSES)": "Peas (Vegetable) (Green)",
            "TAPIOCA": "Tapioca"
        }

        self._load_cost_data()

    def _load_cost_data(self):
        """Load and index cleaned CACP cost dataset."""
        if self.cost_csv_path.exists():
            df = pd.read_csv(self.cost_csv_path)
            # Filter rows with valid C2 cost
            df["cul_cost_c2"] = pd.to_numeric(df["cul_cost_c2"], errors="coerce")
            df["prod_cost_c2"] = pd.to_numeric(df["prod_cost_c2"], errors="coerce")
            df = df[df["cul_cost_c2"].notna() & (df["cul_cost_c2"] > 0)].copy()

            df["state_upper"] = df["state_name"].astype(str).str.strip().str.upper()
            df["crop_upper"] = df["crop_name"].astype(str).str.strip().str.upper()
            # Sort chronologically by survey year so latest records are consistently ordered
            df = df.sort_values("year").copy()
            self.df_cost = df

    def get_cultivation_cost(self, state: str, crop: str, year: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve C2 cultivation cost per hectare and production cost per quintal.
        
        Returns a dictionary with cost values, coverage level, and transparency flags.
        Never fabricates missing costs.
        """
        crop_clean = str(crop).strip()
        crop_upper = crop_clean.upper()
        state_clean = str(state).strip()
        state_upper = state_clean.upper()

        cacp_crop_name = self.crop_to_cacp.get(crop_upper)
        if not cacp_crop_name or self.df_cost is None:
            return {
                "crop": crop_clean,
                "state": state_clean,
                "cul_cost_c2_per_ha": None,
                "prod_cost_c2_per_quintal": None,
                "cost_coverage": "cost_unavailable",
                "is_available": False,
                "is_state_level": False,
                "is_national_benchmark": False,
                "benchmark_year": None,
                "explanation": f"No CACP Cost of Cultivation survey data exists for crop '{crop_clean}'.",
                "disclaimer": "Cost data is unobserved. Profit and ROI cannot be estimated without verified cost figures."
            }

        cacp_crop_upper = cacp_crop_name.upper()

        # 1. Look for State-Level CACP survey records
        state_sub = self.df_cost[
            (self.df_cost["state_upper"] == state_upper) &
            (self.df_cost["crop_upper"] == cacp_crop_upper)
        ]

        if len(state_sub) > 0:
            record = None
            match_type = "latest_state_survey"

            if year is not None:
                # Try exact year match (e.g., '2015-16' or '2015')
                year_str = str(year).strip()
                exact = state_sub[state_sub["year"].astype(str).str.contains(year_str[:4])]
                if len(exact) > 0:
                    record = exact.sort_values("year").iloc[-1]
                    match_type = "exact_year"

            if record is None:
                # Use latest available survey year for this state & crop
                record = state_sub.sort_values("year").iloc[-1]

            cul_c2 = float(record["cul_cost_c2"])
            prod_c2 = float(record["prod_cost_c2"]) if pd.notna(record["prod_cost_c2"]) else None
            rec_year = str(record["year"])

            return {
                "crop": crop_clean,
                "cacp_crop_name": cacp_crop_name,
                "state": state_clean,
                "cul_cost_c2_per_ha": round(cul_c2, 2),
                "prod_cost_c2_per_quintal": round(prod_c2, 2) if prod_c2 else None,
                "cost_coverage": "state_level",
                "is_available": True,
                "is_state_level": True,
                "is_national_benchmark": False,
                "match_type": match_type,
                "benchmark_year": rec_year,
                "explanation": f"Official CACP state survey benchmark for {state_clean} (Survey Year: {rec_year}).",
                "disclaimer": "Cost of cultivation (C2) is surveyed at the state level by CACP and serves as an official regional benchmark, not a district-specific measurement."
            }

        # 2. State has no survey for this crop; fall back to all-India CACP national benchmark with transparent labeling
        nat_sub = self.df_cost[self.df_cost["crop_upper"] == cacp_crop_upper]
        if len(nat_sub) > 0:
            # Average across latest available year for all surveyed states
            latest_year_nat = nat_sub["year"].max()
            nat_latest = nat_sub[nat_sub["year"] == latest_year_nat]
            cul_c2_nat = float(nat_latest["cul_cost_c2"].mean())
            prod_c2_nat = float(nat_latest["prod_cost_c2"].dropna().mean()) if len(nat_latest["prod_cost_c2"].dropna()) > 0 else None

            return {
                "crop": crop_clean,
                "cacp_crop_name": cacp_crop_name,
                "state": state_clean,
                "cul_cost_c2_per_ha": round(cul_c2_nat, 2),
                "prod_cost_c2_per_quintal": round(prod_c2_nat, 2) if prod_c2_nat else None,
                "cost_coverage": "national_benchmark",
                "is_available": True,
                "is_state_level": False,
                "is_national_benchmark": True,
                "match_type": "national_average",
                "benchmark_year": str(latest_year_nat),
                "explanation": f"State {state_clean} is not covered in CACP surveys for {crop_clean}. Using all-India national average benchmark from surveyed states (Survey Year: {latest_year_nat}).",
                "disclaimer": "National benchmark only. Actual costs in this state may differ based on local labor rates, irrigation, and input prices."
            }

        # 3. Completely unobserved
        return {
            "crop": crop_clean,
            "state": state_clean,
            "cul_cost_c2_per_ha": None,
            "prod_cost_c2_per_quintal": None,
            "cost_coverage": "cost_unavailable",
            "is_available": False,
            "is_state_level": False,
            "is_national_benchmark": False,
            "benchmark_year": None,
            "explanation": f"CACP cost data unavailable for {crop_clean}.",
            "disclaimer": "Cost data is unobserved. Profit and ROI cannot be estimated without verified cost figures."
        }

