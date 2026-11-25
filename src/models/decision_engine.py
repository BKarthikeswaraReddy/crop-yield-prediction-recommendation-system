"""
Farmer Decision Support & Crop Recommendation Engine
Machine Learning Based Crop Yield Prediction & Recommendation System

Synthesizes:
  1. ML Production and Yield Predictions (Phase 1 Models)
  2. Mandi Price Intelligence (Historical and Recent APMC Modal Prices)
  3. Cultivation Cost Calculator (CACP State-level C2 Costs)
  4. Financial Calculator (Revenue, Total Cost, Net Profit, ROI)
  5. Weather Intelligence (IMD Seasonal Rainfall and Moisture Risk)

Multi-Criteria Evaluation:
  - Does NOT recommend a crop solely based on high predicted profit.
  - Balances economic return, weather resilience, historical agronomic precedent, and data confidence.
  - Generates transparent, tabular justifications with explicit disclaimers.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.models.predictor import Predictor
from src.models.mandi_intelligence import MandiIntelligence
from src.models.cost_calculator import CultivationCostCalculator
from src.models.financial_calculator import FinancialCalculator
from src.models.weather_intelligence import WeatherIntelligence


class CropRecommendationEngine:

    def __init__(
        self,
        predictor: Optional[Predictor] = None,
        mandi_intel: Optional[MandiIntelligence] = None,
        cost_calc: Optional[CultivationCostCalculator] = None,
        weather_intel: Optional[WeatherIntelligence] = None,
        data_path: Optional[Path] = None
    ):
        base_dir = Path(__file__).resolve().parent.parent.parent
        data_dir = base_dir / "data"

        # Initialize sub-modules
        if predictor is not None:
            self.predictor = predictor
        else:
            csv_path = data_path or (data_dir / "India Agriculture Crop Production.csv")
            self.predictor = Predictor(str(csv_path))

        self.mandi_intel = mandi_intel or MandiIntelligence(data_dir)
        self.cost_calc = cost_calc or CultivationCostCalculator(data_dir)
        self.weather_intel = weather_intel or WeatherIntelligence(data_dir / "weather")

        self.df_prod = self.predictor.data
        self._cache_district_crops()

    def _cache_district_crops(self):
        """Index district-season crop historical frequency."""
        self.district_season_crops = {}
        if self.df_prod is not None:
            grouped = self.df_prod.groupby(["State", "District", "Season"])["Crop"].value_counts()
            for (st, dist, ssn, crp), cnt in grouped.items():
                key = (st.upper(), dist.upper(), ssn.upper())
                if key not in self.district_season_crops:
                    self.district_season_crops[key] = {}
                self.district_season_crops[key][crp] = int(cnt)

    def get_candidate_crops(self, state: str, district: str, season: str) -> List[str]:
        """
        Identify crops historically cultivated in the selected district and season.
        Falls back to district-wide crops or state-wide crops if needed.
        """
        st_u = str(state).strip().upper()
        dist_u = str(district).strip().upper()
        ssn_u = str(season).strip().upper()

        key = (st_u, dist_u, ssn_u)
        if key in self.district_season_crops:
            # Sort by historical record frequency
            sorted_crops = sorted(self.district_season_crops[key].items(), key=lambda x: x[1], reverse=True)
            return [c[0] for c in sorted_crops]

        # Fallback to any season in the district
        if self.df_prod is not None:
            sub = self.df_prod[
                (self.df_prod["State"].str.upper() == st_u) &
                (self.df_prod["District"].str.upper() == dist_u)
            ]
            if len(sub) > 0:
                return sub["Crop"].value_counts().index.tolist()

        return []

    def recommend_crops(
        self,
        state: str,
        district: str,
        season: str,
        area_ha: float,
        year: str = "2022",
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate and rank suitable crops for the selected region, season, and acreage.
        """
        area = max(0.1, float(area_ha))
        candidates = self.get_candidate_crops(state, district, season)

        if not candidates:
            return {
                "state": state,
                "district": district,
                "season": season,
                "area_ha": area,
                "recommendations": [],
                "comparison_table": "No historical crop records found for this state and district combination.",
                "total_evaluated": 0
            }

        evaluated = []

        for crop in candidates:
            # 1. ML Yield Prediction (Phase 1 Yield Model)
            try:
                pred_yield = self.predictor.predict_yield(state, district, crop, year, season, area)
            except Exception as e:
                continue

            if pred_yield <= 0:
                continue

            # Financial calculator formula: Total production = predicted yield * cultivated area
            total_prod = round(pred_yield * area, 4)

            # 2. Mandi Price Intelligence
            price_info = self.mandi_intel.get_price_intelligence(state, district, crop)
            mandi_price = price_info.get("price_per_quintal")

            # 3. Cultivation Cost Calculator
            cost_info = self.cost_calc.get_cultivation_cost(state, crop, year)
            cul_cost = cost_info.get("cul_cost_c2_per_ha")

            # 4. Financial Economics
            fin = FinancialCalculator.calculate(
                predicted_yield_t_ha=pred_yield,
                cultivated_area_ha=area,
                mandi_price_per_quintal=mandi_price,
                cul_cost_c2_per_ha=cul_cost,
                crop_name=crop,
                state_name=state,
                price_level=price_info.get("price_level", "unavailable"),
                cost_level=cost_info.get("cost_coverage", "unavailable")
            )

            # 5. Weather Risk Assessment
            weather = self.weather_intel.assess_weather_risk(state, district, season, crop)

            # 6. Multi-Criteria Scoring (Balances return, risk, and data certainty)
            score = self._compute_composite_score(fin, weather, price_info, cost_info)

            # Generate Rationale Explanation
            rationale = self._generate_crop_rationale(crop, fin, weather, price_info, cost_info)

            evaluated.append({
                "crop": crop,
                "predicted_yield_t_ha": round(pred_yield, 2),
                "predicted_production_tonnes": round(total_prod, 2),
                "mandi_price_per_quintal": mandi_price,
                "mandi_price_level": price_info.get("price_level"),
                "cultivation_cost_per_ha": cul_cost,
                "cost_coverage": cost_info.get("cost_coverage"),
                "gross_revenue": fin.get("gross_revenue"),
                "total_cultivation_cost": fin.get("total_cultivation_cost"),
                "estimated_profit": fin.get("estimated_profit"),
                "roi_percent": fin.get("roi_percent"),
                "financial_status": fin.get("financial_status"),
                "weather_risk_level": weather.get("risk_level"),
                "mean_seasonal_rainfall_mm": weather.get("mean_seasonal_rainfall_mm"),
                "water_requirement": weather.get("water_requirement_category"),
                "composite_score": round(score, 3),
                "rationale": rationale,
                "weather_advisory": weather.get("advisory"),
                "financial_advisory": fin.get("explanation"),
                "price_disclaimer": price_info.get("disclaimer"),
                "cost_disclaimer": cost_info.get("disclaimer")
            })

        # Sort by composite score descending
        evaluated.sort(key=lambda x: x["composite_score"], reverse=True)
        top_recommendations = evaluated[:top_n]

        # Assign ranks
        for idx, rec in enumerate(top_recommendations, start=1):
            rec["rank"] = idx

        # Generate markdown comparison table
        table_md = self._build_comparison_table(top_recommendations)

        return {
            "state": state,
            "district": district,
            "season": season,
            "area_ha": area,
            "total_evaluated": len(evaluated),
            "recommendations": top_recommendations,
            "comparison_table": table_md
        }

    def _compute_composite_score(
        self,
        fin: Dict[str, Any],
        weather: Dict[str, Any],
        price_info: Dict[str, Any],
        cost_info: Dict[str, Any]
    ) -> float:
        """
        Compute multi-criteria suitability score.
        Ensures a crop is not ranked #1 solely due to high predicted profit.
        """
        # A. Economic return component (weight: 35%)
        profit = fin.get("estimated_profit")
        roi = fin.get("roi_percent")
        econ_score = 0.5  # neutral default

        if profit is not None and roi is not None:
            if profit > 0:
                # Sigmoid-like scaling for ROI (100% ROI -> ~0.9)
                roi_clamped = min(200.0, max(-100.0, roi))
                econ_score = 0.5 + 0.5 * (roi_clamped / (abs(roi_clamped) + 50.0))
            else:
                econ_score = max(0.05, 0.5 + (profit / (abs(profit) + 50000.0)) * 0.45)
        elif fin.get("gross_revenue") is not None:
            econ_score = 0.55  # revenue available but cost unknown

        # B. Weather resilience component (weight: 30%)
        # Lower risk -> higher score
        risk_score = weather.get("risk_score", 0.5)
        weather_resilience = 1.0 - risk_score

        # C. Data Confidence component (weight: 20%)
        # Verified district price & state cost get highest confidence
        conf_score = 0.4
        if price_info.get("price_level") == "district":
            conf_score += 0.3
        elif price_info.get("price_level") == "state":
            conf_score += 0.15

        if cost_info.get("cost_coverage") == "state_level":
            conf_score += 0.3
        elif cost_info.get("cost_coverage") == "national_benchmark":
            conf_score += 0.15

        # D. Yield reliability component (weight: 15%)
        yield_val = fin.get("predicted_yield_t_ha", 1.0)
        yield_score = min(1.0, max(0.2, yield_val / 5.0))

        # Composite weighted sum
        composite = (
            0.35 * econ_score +
            0.30 * weather_resilience +
            0.20 * conf_score +
            0.15 * yield_score
        )

        return float(composite)

    def _generate_crop_rationale(
        self,
        crop: str,
        fin: Dict[str, Any],
        weather: Dict[str, Any],
        price_info: Dict[str, Any],
        cost_info: Dict[str, Any]
    ) -> str:
        """Construct detailed textual explanation for the crop recommendation."""
        reasons = []

        # Yield
        y_val = fin.get("predicted_yield_t_ha", 0.0)
        reasons.append(f"Predicted yield of {y_val:.2f} t/ha.")

        # Economics
        profit = fin.get("estimated_profit")
        roi = fin.get("roi_percent")
        if profit is not None and roi is not None:
            if profit >= 0:
                reasons.append(f"Estimated profit of ₹{profit:,.2f} with an ROI of {roi:+.1f}%.")
            else:
                reasons.append(f"Projected loss of -₹{abs(profit):,.2f} under standard cost structures.")
        elif fin.get("gross_revenue") is not None:
            reasons.append(f"Estimated gross revenue of ₹{fin['gross_revenue']:,.2f} (CACP cultivation cost unobserved).")
        else:
            reasons.append("Market price and cultivation cost are currently unobserved.")

        # Weather Risk
        w_risk = weather.get("risk_level", "Unrated")
        rf_mm = weather.get("mean_seasonal_rainfall_mm")
        if rf_mm is not None:
            reasons.append(f"{w_risk} weather risk with {rf_mm:.0f} mm seasonal rainfall.")
        else:
            reasons.append(f"Weather risk: {w_risk}.")

        # Data Confidence
        p_lvl = price_info.get("price_level")
        c_lvl = cost_info.get("cost_coverage")
        if p_lvl == "district" and c_lvl == "state_level":
            reasons.append("High data certainty (district APMC prices & CACP state cost survey).")
        elif p_lvl == "state":
            reasons.append("Moderate certainty (state-level price benchmark).")

        return " ".join(reasons)

    def _build_comparison_table(self, recommendations: List[Dict[str, Any]]) -> str:
        """Render a formatted markdown comparison table."""
        if not recommendations:
            return "No recommendations available."

        headers = [
            "Rank",
            "Crop",
            "Predicted Yield (t/ha)",
            "Total Prod (Tonnes)",
            "Modal Price (₹/q)",
            "Price Source",
            "Cost (₹/ha)",
            "Est. Revenue (₹)",
            "Est. Profit (₹)",
            "ROI (%)",
            "Weather Risk",
            "Overall Evaluation / Advisory"
        ]

        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"
        ]

        for r in recommendations:
            price_str = f"₹{r['mandi_price_per_quintal']:,.0f}" if r["mandi_price_per_quintal"] else "N/A"
            cost_str = f"₹{r['cultivation_cost_per_ha']:,.0f}" if r["cultivation_cost_per_ha"] else "N/A"
            rev_str = f"₹{r['gross_revenue']:,.0f}" if r["gross_revenue"] is not None else "N/A"
            profit_str = f"₹{r['estimated_profit']:,.0f}" if r["estimated_profit"] is not None else "N/A"
            roi_str = f"{r['roi_percent']:+.1f}%" if r["roi_percent"] is not None else "N/A"

            row = [
                f"**#{r['rank']}**",
                f"**{r['crop']}**",
                f"{r['predicted_yield_t_ha']:.2f}",
                f"{r['predicted_production_tonnes']:,.1f}",
                price_str,
                r["mandi_price_level"].capitalize() if r["mandi_price_level"] else "N/A",
                cost_str,
                rev_str,
                profit_str,
                roi_str,
                r["weather_risk_level"],
                r["rationale"]
            ]
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
