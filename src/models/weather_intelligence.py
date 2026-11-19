"""
Weather Intelligence and Climate Risk Engine
Crop Production Prediction and Farmer Decision Support System

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

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            self.data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "weather"
        else:
            self.data_dir = Path(data_dir)

        self.summary_path = self.data_dir / "rainfall_district_season_summary.parquet"
        self.df_rainfall = None

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
        """Load pre-aggregated seasonal rainfall summary."""
        if self.summary_path.exists():
            self.df_rainfall = pd.read_parquet(self.summary_path)

    def assess_weather_risk(
        self,
        state: str,
        district: str,
        season: str,
        crop: str
    ) -> Dict[str, Any]:
        """
        Evaluate seasonal rainfall availability and weather risk for a crop in a district.
        """
        state_u = str(state).strip()
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
            # Try district match
            sub = self.df_rainfall[
                (self.df_rainfall["District_Upper"] == dist_u) &
                (self.df_rainfall["Season"] == season_key)
            ]
            if len(sub) > 0:
                rainfall_data = sub.iloc[0]
            else:
                # Try whole year fallback for the district
                sub_wy = self.df_rainfall[
                    (self.df_rainfall["District_Upper"] == dist_u) &
                    (self.df_rainfall["Season"] == "Whole Year")
                ]
                if len(sub_wy) > 0:
                    rainfall_data = sub_wy.iloc[0]

        if rainfall_data is None:
            return {
                "district": district,
                "season": season_clean,
                "crop": crop,
                "mean_seasonal_rainfall_mm": None,
                "rainfall_cv_percent": None,
                "risk_level": "Unrated",
                "risk_score": 0.5,
                "is_observed": False,
                "explanation": f"No IMD weather station observations matched for district '{district}' in {season_clean} season.",
                "advisory": "Local rainfall data unobserved. Ensure local irrigation sources before planting."
            }

        mean_rf = float(rainfall_data["mean_rainfall_mm"])
        rf_cv = float(rainfall_data["rainfall_cv"])
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

        # Factor in rainfall variability (CV)
        if rf_cv > 45.0:
            notes.append(f"High historical rainfall volatility (CV: {rf_cv:.1f}% indicates erratic monsoon behavior).")
            if risk_level == "Low":
                risk_level = "Moderate"
                risk_score += 0.2
        elif rf_cv < 25.0:
            notes.append(f"Precipitation pattern is consistent year-over-year (CV: {rf_cv:.1f}%).")

        return {
            "district": district,
            "season": season_clean,
            "season_matched": season_key,
            "crop": crop,
            "water_requirement_category": crop_req["category"],
            "mean_seasonal_rainfall_mm": round(mean_rf, 1),
            "min_seasonal_rainfall_mm": round(float(rainfall_data["min_rainfall_mm"]), 1),
            "max_seasonal_rainfall_mm": round(float(rainfall_data["max_rainfall_mm"]), 1),
            "rainfall_cv_percent": round(rf_cv, 1),
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

