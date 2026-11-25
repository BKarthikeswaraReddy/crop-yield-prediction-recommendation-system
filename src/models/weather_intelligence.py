"""
Weather Intelligence and Climate Risk Engine
Machine Learning Based Crop Yield Prediction & Recommendation System

Analyzes historical IMD district-wise rainfall data and evaluates crop-specific
seasonal water requirements, precipitation variability, and moisture risks.

Important Principles:
- Uses verified IMD historical rainfall observations.
- Never fabricates weather data.
- Accurately reports missing weather coverage.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class WeatherIntelligence:

    STATE_ALIASES = {
        "ANDAMAN & NICOBAR": "ANDAMAN AND NICOBAR ISLANDS",
        "ANDAMAN & NICOBAR ISLANDS": "ANDAMAN AND NICOBAR ISLANDS",
        "CHHATISGARH": "CHHATTISGARH",
        "DADRA & NAGAR HAVELI": "DADRA AND NAGAR HAVELI",
        "DAMAN & DIU": "DAMAN AND DIU",
        "JAMMU & KASHMIR": "JAMMU AND KASHMIR",
        "LADDAKH": "LADAKH",
        "NCT OF DELHI": "DELHI",
        "ORISSA": "ODISHA",
        "PONDICHERRY": "PUDUCHERRY",
        "TAMILNADU": "TAMIL NADU",
        "UTTARANCHAL": "UTTARAKHAND",
    }

    # Approximate State/UT bounding boxes (lat_min, lat_max, lon_min, lon_max) for
    # State-level aggregation of the 1°×1° IMD temperature grid (2018-2025).
    # District-level mapping is intentionally omitted without official district shapefiles.
    STATE_BBOX = {
        "ANDHRA PRADESH": (12.6, 19.1, 76.8, 84.8),
        "ARUNACHAL PRADESH": (26.6, 29.5, 91.5, 97.4),
        "ASSAM": (24.1, 28.0, 89.7, 96.0),
        "BIHAR": (24.3, 27.5, 83.3, 88.3),
        "CHHATTISGARH": (17.8, 24.1, 80.2, 84.4),
        "GOA": (14.9, 15.8, 73.7, 74.4),
        "GUJARAT": (20.1, 24.7, 68.1, 74.5),
        "HARYANA": (27.6, 30.9, 74.5, 77.6),
        "HIMACHAL PRADESH": (30.4, 33.2, 75.6, 79.0),
        "JAMMU AND KASHMIR": (32.3, 35.0, 73.8, 76.5),
        "JHARKHAND": (21.9, 25.3, 83.3, 87.9),
        "KARNATAKA": (11.6, 18.5, 74.0, 78.6),
        "KERALA": (8.2, 12.8, 74.8, 77.4),
        "LADAKH": (32.5, 36.5, 75.8, 79.5),
        "MADHYA PRADESH": (21.1, 26.9, 74.0, 82.8),
        "MAHARASHTRA": (15.6, 22.0, 72.6, 80.9),
        "MANIPUR": (23.8, 25.7, 93.0, 94.8),
        "MEGHALAYA": (25.0, 26.1, 89.8, 92.8),
        "MIZORAM": (21.9, 24.5, 92.2, 93.4),
        "NAGALAND": (25.2, 27.0, 93.3, 95.2),
        "ODISHA": (17.8, 22.6, 81.4, 87.5),
        "PUNJAB": (29.5, 32.5, 73.9, 76.9),
        "RAJASTHAN": (23.0, 30.2, 69.5, 78.3),
        "SIKKIM": (27.1, 28.1, 88.0, 88.9),
        "TAMIL NADU": (8.1, 13.6, 76.2, 80.3),
        "TELANGANA": (15.8, 19.9, 77.2, 81.3),
        "TRIPURA": (22.9, 24.5, 91.1, 92.3),
        "UTTAR PRADESH": (23.9, 30.4, 77.1, 84.6),
        "UTTARAKHAND": (28.7, 31.5, 77.6, 81.0),
        "WEST BENGAL": (21.5, 27.2, 85.8, 89.9),
    }

    SEASON_MONTHS = {
        "Kharif": [6, 7, 8, 9, 10],
        "Rabi": [11, 12, 1, 2, 3],
        "Summer": [3, 4, 5, 6],
        "Autumn": [9, 10, 11],
        "Winter": [12, 1, 2],
        "Whole Year": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    }

    COOL_SEASON_CROPS = {
        "WHEAT", "GRAM", "CHICKPEAS", "BARLEY", "MUSTARD",
        "RAPESEED &MUSTARD", "POTATO", "MASOOR", "LENTIL",
    }

    @classmethod
    def normalize_state_name(cls, state: str) -> str:
        """Normalize state names consistently for reliable state-district matching."""
        cleaned = " ".join(str(state).strip().upper().split())
        return cls.STATE_ALIASES.get(cleaned, cleaned)

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            self.data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "weather"
        else:
            self.data_dir = Path(data_dir)

        self.summary_path = self.data_dir / "rainfall_district_season_summary.parquet"
        self.temp_summary_path = self.data_dir / "temperature_monthly_summary.parquet"
        self.df_rainfall = None
        self.df_temperature = None

        # Agronomic water requirements (approximate seasonal water need in mm) based on ICAR guidelines
        self.crop_water_requirements = {
            # High water (> 900 mm)
            "RICE": {"category": "high", "min_mm": 900, "ideal_mm": 1300, "drought_sensitive": True},
            "PADDY": {"category": "high", "min_mm": 900, "ideal_mm": 1300, "drought_sensitive": True},
            "SUGARCANE": {"category": "high", "min_mm": 1200, "ideal_mm": 1800, "drought_sensitive": True},
            "JUTE": {"category": "high", "min_mm": 800, "ideal_mm": 1200, "drought_sensitive": True},

            # Moderate water (400 - 850 mm)
            "WHEAT": {"category": "moderate", "min_mm": 350, "ideal_mm": 550, "drought_sensitive": False},
            "MAIZE": {"category": "moderate", "min_mm": 450, "ideal_mm": 700, "drought_sensitive": True},
            "CORN": {"category": "moderate", "min_mm": 450, "ideal_mm": 700, "drought_sensitive": True},
            "COTTON": {"category": "moderate", "min_mm": 500, "ideal_mm": 800, "drought_sensitive": False},
            "COTTON(LINT)": {"category": "moderate", "min_mm": 500, "ideal_mm": 800, "drought_sensitive": False},
            "GROUNDNUT": {"category": "moderate", "min_mm": 450, "ideal_mm": 650, "drought_sensitive": False},
            "SOYABEAN": {"category": "moderate", "min_mm": 450, "ideal_mm": 700, "drought_sensitive": False},
            "SOYBEAN": {"category": "moderate", "min_mm": 450, "ideal_mm": 700, "drought_sensitive": False},
            "POTATO": {"category": "moderate", "min_mm": 400, "ideal_mm": 600, "drought_sensitive": True},
            "ONION": {"category": "moderate", "min_mm": 350, "ideal_mm": 550, "drought_sensitive": True},
            "SUNFLOWER": {"category": "moderate", "min_mm": 350, "ideal_mm": 550, "drought_sensitive": False},

            # Low water / drought hardy (< 450 mm)
            "BAJRA": {"category": "low", "min_mm": 250, "ideal_mm": 400, "drought_sensitive": False},
            "JOWAR": {"category": "low", "min_mm": 300, "ideal_mm": 450, "drought_sensitive": False},
            "RAGI": {"category": "low", "min_mm": 300, "ideal_mm": 450, "drought_sensitive": False},
            "GRAM": {"category": "low", "min_mm": 200, "ideal_mm": 350, "drought_sensitive": False},
            "CHICKPEAS": {"category": "low", "min_mm": 200, "ideal_mm": 350, "drought_sensitive": False},
            "ARHAR/TUR": {"category": "low", "min_mm": 300, "ideal_mm": 450, "drought_sensitive": False},
            "TUR": {"category": "low", "min_mm": 300, "ideal_mm": 450, "drought_sensitive": False},
            "MOONG(GREEN GRAM)": {"category": "low", "min_mm": 250, "ideal_mm": 380, "drought_sensitive": False},
            "MOONG": {"category": "low", "min_mm": 250, "ideal_mm": 380, "drought_sensitive": False},
            "URAD": {"category": "low", "min_mm": 250, "ideal_mm": 380, "drought_sensitive": False},
            "MASOOR": {"category": "low", "min_mm": 200, "ideal_mm": 350, "drought_sensitive": False},
            "BARLEY": {"category": "low", "min_mm": 200, "ideal_mm": 350, "drought_sensitive": False},
            "RAPESEED &MUSTARD": {"category": "low", "min_mm": 200, "ideal_mm": 350, "drought_sensitive": False},
            "MUSTARD": {"category": "low", "min_mm": 200, "ideal_mm": 350, "drought_sensitive": False},
            "SESAMUM": {"category": "low", "min_mm": 250, "ideal_mm": 400, "drought_sensitive": False},
            "SAFFLOWER": {"category": "low", "min_mm": 200, "ideal_mm": 350, "drought_sensitive": False},
            "NIGER SEED": {"category": "low", "min_mm": 250, "ideal_mm": 400, "drought_sensitive": False},
        }

        self._load_summary()

    def _load_summary(self):
        """Load pre-aggregated seasonal rainfall and monthly gridded temperature summaries."""
        if self.summary_path.exists():
            df = pd.read_parquet(self.summary_path)
            if "State_Clean" in df.columns:
                df["State_Upper"] = df["State_Clean"].map(self.normalize_state_name)
            else:
                df["State_Upper"] = ""
            df["District_Upper"] = df["District_Upper"].astype(str).str.strip().str.upper()
            self.df_rainfall = df

        if self.temp_summary_path.exists():
            self.df_temperature = pd.read_parquet(self.temp_summary_path)

    def get_state_temperature_profile(
        self,
        state: Optional[str],
        season: str = "Kharif",
        crop: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute State-level seasonal temperature and heat-stress indicators from
        the 1°×1° IMD monthly gridded temperature summary (2018-2025).

        Honesty guarantee: Never fabricates temperature observations for states/UTs
        that lack 1°×1° grid coverage, and explicitly labels spatial resolution as
        state-level gridded benchmark (not district weather stations).
        """
        state_u = self.normalize_state_name(state) if state and str(state).strip() else ""
        season_clean = str(season).strip()
        season_key = "Whole Year"
        for s in ["Kharif", "Rabi", "Summer", "Autumn", "Winter"]:
            if s.lower() in season_clean.lower():
                season_key = s
                break

        unavailable = {
            "state": state or "",
            "season": season_clean,
            "season_matched": season_key,
            "is_observed": False,
            "spatial_resolution": "state_1deg_grid",
            "mean_max_temp_c": None,
            "mean_min_temp_c": None,
            "peak_max_temp_c": None,
            "lowest_min_temp_c": None,
            "grid_points_matched": 0,
            "years_observed": 0,
            "heat_stress_level": "Unobserved",
            "explanation": (
                f"No 1°×1° IMD temperature grid coverage mapped for '{state or 'unspecified state'}'. "
                "District-level temperature stations are not available in this dataset."
            ),
            "disclaimer": (
                "IMD temperature data is a 1°×1° (~111 km) spatial grid benchmark (2018–2025) "
                "and is not fabricated where coverage is absent."
            ),
        }

        if self.df_temperature is None or not state_u or state_u not in self.STATE_BBOX:
            return unavailable

        lat_min, lat_max, lon_min, lon_max = self.STATE_BBOX[state_u]
        months = self.SEASON_MONTHS.get(season_key, self.SEASON_MONTHS["Whole Year"])

        sub = self.df_temperature[
            (self.df_temperature["Latitude"] >= lat_min)
            & (self.df_temperature["Latitude"] <= lat_max)
            & (self.df_temperature["Longitude"] >= lon_min)
            & (self.df_temperature["Longitude"] <= lon_max)
            & (self.df_temperature["Month"].isin(months))
        ]

        if len(sub) == 0:
            return unavailable

        mean_max = float(sub["Avg_Max_Temp_C"].mean())
        mean_min = float(sub["Avg_Min_Temp_C"].mean())
        peak_max = float(sub["Max_of_Max_Temp_C"].max())
        lowest_min = float(sub["Min_of_Min_Temp_C"].min())
        grid_pts = int(sub[["Latitude", "Longitude"]].drop_duplicates().shape[0])
        years_obs = int(sub["Year"].nunique())

        crop_u = str(crop).strip().upper() if crop else ""
        is_cool_crop = crop_u in self.COOL_SEASON_CROPS or season_key in ("Rabi", "Winter")

        if is_cool_crop:
            if peak_max >= 36.0 or mean_max >= 32.0:
                stress = "High"
                note = (
                    f"Elevated terminal heat-stress risk for cool-season/Rabi crops "
                    f"(mean max {mean_max:.1f}°C, peak {peak_max:.1f}°C)."
                )
            elif mean_max >= 28.0:
                stress = "Moderate"
                note = (
                    f"Moderate thermal load for Rabi crops (mean max {mean_max:.1f}°C, "
                    f"min {mean_min:.1f}°C); timely sowing mitigates late-season heat."
                )
            else:
                stress = "Low"
                note = (
                    f"Favorable cool-season thermal regime (mean max {mean_max:.1f}°C, "
                    f"mean min {mean_min:.1f}°C)."
                )
        else:
            if peak_max >= 42.0 or mean_max >= 38.0:
                stress = "High"
                note = (
                    f"Severe heat-stress potential during peak season "
                    f"(mean max {mean_max:.1f}°C, peak {peak_max:.1f}°C)."
                )
            elif mean_max >= 34.0 or peak_max >= 39.0:
                stress = "Moderate"
                note = (
                    f"Warm seasonal temperatures (mean max {mean_max:.1f}°C, peak {peak_max:.1f}°C); "
                    "monitor soil moisture during dry spells."
                )
            else:
                stress = "Low"
                note = (
                    f"Moderate thermal profile (mean max {mean_max:.1f}°C, mean min {mean_min:.1f}°C)."
                )

        return {
            "state": state_u,
            "season": season_clean,
            "season_matched": season_key,
            "is_observed": True,
            "spatial_resolution": "state_1deg_grid",
            "mean_max_temp_c": round(mean_max, 1),
            "mean_min_temp_c": round(mean_min, 1),
            "peak_max_temp_c": round(peak_max, 1),
            "lowest_min_temp_c": round(lowest_min, 1),
            "grid_points_matched": grid_pts,
            "years_observed": years_obs,
            "heat_stress_level": stress,
            "explanation": note,
            "disclaimer": (
                f"State-level IMD 1°×1° gridded temperature benchmark ({grid_pts} grid cells, "
                f"{years_obs} years: 2018–2025). District-specific weather stations are not mapped "
                "without official district boundary shapefiles."
            ),
        }

    def get_district_seasonal_history(self, state: Optional[str], district: str) -> list:
        """Return seasonal rainfall history rows for a district, disambiguated by state."""
        if self.df_rainfall is None or not district:
            return []
        dist_u = str(district).strip().upper()
        state_u = self.normalize_state_name(state) if state and str(state).strip() else ""

        if state_u:
            sub = self.df_rainfall[
                (self.df_rainfall["State_Upper"] == state_u) &
                (self.df_rainfall["District_Upper"] == dist_u)
            ]
            if len(sub) > 0:
                return sub.to_dict(orient="records")
            # Do not leak another state's district when state was explicitly provided
            other_states = self.df_rainfall[self.df_rainfall["District_Upper"] == dist_u]["State_Upper"].unique()
            if len(other_states) > 0:
                return []

        sub = self.df_rainfall[self.df_rainfall["District_Upper"] == dist_u]
        if len(sub) > 0 and sub["State_Upper"].nunique() == 1:
            return sub.to_dict(orient="records")
        return []

    def assess_weather_risk(
        self,
        state: str,
        district: str,
        season: str,
        crop: str
    ) -> Dict[str, Any]:
        """
        Evaluate seasonal rainfall availability and weather risk for a crop in a district.
        Disambiguates identically named districts across different states using State_Upper.
        """
        state_u = self.normalize_state_name(state) if state and str(state).strip() else ""
        dist_u = str(district).strip().upper()
        season_clean = str(season).strip()
        crop_u = str(crop).strip().upper()

        # Map common season names to summary season keys
        season_key = "Whole Year"
        for s in ["Kharif", "Rabi", "Summer", "Autumn", "Winter"]:
            if s.lower() in season_clean.lower():
                season_key = s
                break

        rainfall_data = None
        if self.df_rainfall is not None:
            # 1. Exact state + district match (prevents cross-state collisions like Aurangabad)
            if state_u:
                state_dist_sub = self.df_rainfall[
                    (self.df_rainfall["State_Upper"] == state_u) &
                    (self.df_rainfall["District_Upper"] == dist_u)
                ]
                if len(state_dist_sub) > 0:
                    sub = state_dist_sub[state_dist_sub["Season"] == season_key]
                    if len(sub) > 0:
                        rainfall_data = sub.sort_values("years_observed", ascending=False).iloc[0]
                    else:
                        sub_wy = state_dist_sub[state_dist_sub["Season"] == "Whole Year"]
                        if len(sub_wy) > 0:
                            rainfall_data = sub_wy.sort_values("years_observed", ascending=False).iloc[0]
            else:
                # 2. District-only lookup only when unambiguous (single state owns district)
                dist_sub = self.df_rainfall[self.df_rainfall["District_Upper"] == dist_u]
                if len(dist_sub) > 0 and dist_sub["State_Upper"].nunique() == 1:
                    sub = dist_sub[dist_sub["Season"] == season_key]
                    if len(sub) > 0:
                        rainfall_data = sub.sort_values("years_observed", ascending=False).iloc[0]
                    else:
                        sub_wy = dist_sub[dist_sub["Season"] == "Whole Year"]
                        if len(sub_wy) > 0:
                            rainfall_data = sub_wy.sort_values("years_observed", ascending=False).iloc[0]

        if rainfall_data is None:
            return {
                "state": state,
                "district": district,
                "season": season_clean,
                "crop": crop,
                "mean_seasonal_rainfall_mm": None,
                "rainfall_cv_percent": None,
                "risk_level": "Unrated",
                "risk_score": 0.5,
                "is_observed": False,
                "explanation": f"No IMD weather station observations matched for district '{district}' ({state or 'unspecified state'}) in {season_clean} season.",
                "advisory": "Local rainfall data unobserved. Ensure local irrigation sources before planting."
            }

        mean_rf = float(rainfall_data["mean_rainfall_mm"])
        raw_cv = rainfall_data["rainfall_cv"]
        rf_cv = float(raw_cv) if pd.notna(raw_cv) else None
        crop_req = self.crop_water_requirements.get(crop_u, {"category": "moderate", "min_mm": 400, "ideal_mm": 600, "drought_sensitive": False})

        # Evaluate risk level based on crop water requirements vs seasonal rainfall
        risk_level = "Low"
        risk_score = 0.2  # 0.0 (safest) to 1.0 (highest risk)
        notes = []

        if crop_req["category"] == "high":
            if mean_rf < crop_req["min_mm"]:
                risk_level = "High"
                risk_score = 0.85
                notes.append(f"Seasonal rainfall ({mean_rf:.0f} mm) is well below the crop's minimal requirement ({crop_req['min_mm']} mm). Critical dependency on irrigation.")
            elif mean_rf < crop_req["ideal_mm"]:
                risk_level = "Moderate"
                risk_score = 0.55
                notes.append(f"Seasonal rainfall ({mean_rf:.0f} mm) is moderate. Supplementary irrigation required during dry spells.")
            else:
                risk_level = "Low"
                risk_score = 0.25
                notes.append(f"Plentiful rainfall ({mean_rf:.0f} mm) fully covers crop moisture requirements.")

        elif crop_req["category"] == "moderate":
            if mean_rf < 200 and season_key != "Rabi":
                risk_level = "High"
                risk_score = 0.75
                notes.append(f"Low seasonal precipitation ({mean_rf:.0f} mm). Drought stress risk.")
            elif mean_rf < crop_req["min_mm"]:
                risk_level = "Moderate"
                risk_score = 0.45
                notes.append(f"Rainfall ({mean_rf:.0f} mm) is slightly below ideal; manageable with standard irrigation.")
            else:
                risk_level = "Low"
                risk_score = 0.20
                notes.append(f"Rainfall ({mean_rf:.0f} mm) is adequate for normal growth.")

        else:  # low water / drought-hardy
            if mean_rf < 100:
                risk_level = "Moderate"
                risk_score = 0.40
                notes.append(f"Extremely dry season ({mean_rf:.0f} mm), but crop is naturally drought-tolerant.")
            else:
                risk_level = "Low"
                risk_score = 0.15
                notes.append(f"Drought-hardy crop well-suited to seasonal rainfall ({mean_rf:.0f} mm).")

        # Factor in rainfall variability (CV) when multi-year observations exist
        if rf_cv is not None:
            if rf_cv > 45.0:
                notes.append(f"High historical rainfall volatility (CV: {rf_cv:.1f}% indicates erratic monsoon behavior).")
                if risk_level == "Low":
                    risk_level = "Moderate"
                    risk_score += 0.2
            elif rf_cv < 25.0:
                notes.append(f"Precipitation pattern is consistent year-over-year (CV: {rf_cv:.1f}%).")

        return {
            "state": str(rainfall_data["State_Clean"]),
            "district": district,
            "season": season_clean,
            "season_matched": season_key,
            "crop": crop,
            "water_requirement_category": crop_req["category"],
            "mean_seasonal_rainfall_mm": round(mean_rf, 1),
            "min_seasonal_rainfall_mm": round(float(rainfall_data["min_rainfall_mm"]), 1),
            "max_seasonal_rainfall_mm": round(float(rainfall_data["max_rainfall_mm"]), 1),
            "rainfall_cv_percent": round(rf_cv, 1) if rf_cv is not None else None,
            "risk_level": risk_level,
            "risk_score": round(risk_score, 2),
            "is_observed": True,
            "years_observed": int(rainfall_data["years_observed"]),
            "explanation": " ".join(notes),
            "advisory": (
                "Ensure assured irrigation channel or borewell support if planting moisture-sensitive crops under elevated risk."
                if risk_level in ["Moderate", "High"] else "Weather risk is favorable for planting."
            )
        }

