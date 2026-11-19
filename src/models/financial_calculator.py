"""
Financial Calculator
Crop Production Prediction and Farmer Decision Support System

Calculates complete farm economics:
  - Total production = predicted yield × cultivated area (tonnes)
  - Production in quintals = total production × 10
  - Gross revenue = production in quintals × estimated mandi price per quintal (₹)
  - Total cultivation cost = cultivation cost per hectare × cultivated area (₹)
  - Estimated profit = gross revenue − total cultivation cost (₹)
  - ROI (%) = (profit / total cultivation cost) × 100, when cost is valid and positive

Important Principles:
  - Never fabricate missing price or cost values.
  - If price or cost is unavailable, profit and ROI are labeled unavailable (None) with clear explanations.
  - Transparently returns all component terms, units, and conversion factors.
"""

from typing import Dict, Any, Optional


class FinancialCalculator:

    @staticmethod
    def calculate(
        predicted_yield_t_ha: float,
        cultivated_area_ha: float,
        mandi_price_per_quintal: Optional[float] = None,
        cul_cost_c2_per_ha: Optional[float] = None,
        crop_name: str = "",
        state_name: str = "",
        price_level: str = "district",
        cost_level: str = "state_level"
    ) -> Dict[str, Any]:
        """
        Calculate total production, revenue, total cost, net profit, and ROI.
        """
        area = max(0.0, float(cultivated_area_ha))
        yield_val = max(0.0, float(predicted_yield_t_ha))

        # 1. Total Production
        total_prod_tonnes = round(yield_val * area, 2)
        total_prod_quintals = round(total_prod_tonnes * 10.0, 2)  # 1 metric tonne = exactly 10 quintals

        # 2. Gross Revenue
        gross_revenue = None
        has_price = mandi_price_per_quintal is not None and float(mandi_price_per_quintal) > 0
        if has_price:
            price = float(mandi_price_per_quintal)
            gross_revenue = round(total_prod_quintals * price, 2)

        # 3. Total Cultivation Cost
        total_cost = None
        has_cost = cul_cost_c2_per_ha is not None and float(cul_cost_c2_per_ha) > 0
        if has_cost:
            cost_ha = float(cul_cost_c2_per_ha)
            total_cost = round(cost_ha * area, 2)

        # 4. Net Profit & ROI
        net_profit = None
        roi_percent = None
        financial_status = "complete"

        if has_price and has_cost:
            net_profit = round(gross_revenue - total_cost, 2)
            if total_cost > 0:
                roi_percent = round((net_profit / total_cost) * 100.0, 2)
            else:
                roi_percent = 0.0
        elif has_price and not has_cost:
            financial_status = "cost_unavailable"
        elif not has_price and has_cost:
            financial_status = "price_unavailable"
        else:
            financial_status = "price_and_cost_unavailable"

        # Generate clear advisory summary
        explanation = []
        if has_price:
            explanation.append(f"Gross revenue: ₹{gross_revenue:,.2f} based on {price_level} modal price of ₹{mandi_price_per_quintal:,.2f}/quintal.")
        else:
            explanation.append("Mandi price unobserved; gross revenue cannot be calculated.")

        if has_cost:
            explanation.append(f"Total cost: ₹{total_cost:,.2f} based on {cost_level} benchmark of ₹{cul_cost_c2_per_ha:,.2f}/ha.")
        else:
            explanation.append("CACP cultivation cost unobserved; total cost cannot be calculated.")

        if net_profit is not None:
            if net_profit >= 0:
                explanation.append(f"Estimated net profit: ₹{net_profit:,.2f} (ROI: {roi_percent:+.1f}%).")
            else:
                explanation.append(f"Estimated loss: -₹{abs(net_profit):,.2f} (ROI: {roi_percent:.1f}%).")

        return {
            "crop": crop_name,
            "state": state_name,
            "cultivated_area_ha": round(area, 2),
            "predicted_yield_t_ha": round(yield_val, 2),
            "total_production_tonnes": round(total_prod_tonnes, 2),
            "total_production_quintals": round(total_prod_quintals, 2),
            "mandi_price_per_quintal": round(float(mandi_price_per_quintal), 2) if has_price else None,
            "cul_cost_c2_per_ha": round(float(cul_cost_c2_per_ha), 2) if has_cost else None,
            "gross_revenue": gross_revenue,
            "total_cultivation_cost": total_cost,
            "estimated_profit": net_profit,
            "roi_percent": roi_percent,
            "financial_status": financial_status,
            "price_level": price_level,
            "cost_level": cost_level,
            "explanation": " ".join(explanation),
            "disclaimer": (
                "Financial projections are planning benchmarks. Actual farmer realizations depend on "
                "local grading, transport charges, input variation, and intra-season price volatility."
            )
        }

