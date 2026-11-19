import os
import sys
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = ROOT_DIR

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

LEGACY_DIR = os.path.join(ROOT_DIR, "crop_Production_prediction_new", "crop-production-predictor")
if os.path.exists(LEGACY_DIR) and LEGACY_DIR not in sys.path:
    sys.path.insert(1, LEGACY_DIR)

from src.models.mandi_intelligence import MandiIntelligence
from src.models.cost_calculator import CultivationCostCalculator
from src.models.financial_calculator import FinancialCalculator
from src.models.weather_intelligence import WeatherIntelligence
from src.models.decision_engine import CropRecommendationEngine


@pytest.fixture(scope="module")
def mandi_intel():
    return MandiIntelligence()


@pytest.fixture(scope="module")
def cost_calc():
    return CultivationCostCalculator()


@pytest.fixture(scope="module")
def weather_intel():
    return WeatherIntelligence()


@pytest.fixture(scope="module")
def recommendation_engine():
    return CropRecommendationEngine()


# -----------------------------------------------------------------------------
# 1. Mandi Price Intelligence Tests
# -----------------------------------------------------------------------------

def test_mandi_district_level_match(mandi_intel):
    """Verify district-level modal price matching with verified APMC transactions."""
    res = mandi_intel.get_price_intelligence(state="Punjab", district="Ludhiana", crop="Wheat")
    assert res["price_per_quintal"] is not None
    assert res["price_per_quintal"] > 0
    assert res["price_level"] == "district"
    assert res["is_estimate"] is False
    assert res["confidence"] == "high"
    assert len(res["markets"]) > 0


def test_mandi_state_level_fallback(mandi_intel):
    """Verify state-level benchmark fallback when district has no direct mandi records."""
    res = mandi_intel.get_price_intelligence(state="Assam", district="Baksa", crop="Wheat")
    assert res["price_per_quintal"] is not None
    assert res["price_level"] == "state"
    assert res["is_estimate"] is True
    assert res["is_state_benchmark"] is True
    assert res["confidence"] == "medium"


def test_mandi_unavailable_crop(mandi_intel):
    """Verify that unobserved crops report unavailable status without fabricating prices."""
    res = mandi_intel.get_price_intelligence(state="Kerala", district="Idukki", crop="NonExistentCrop123")
    assert res["price_per_quintal"] is None
    assert res["price_level"] == "unavailable"
    assert res["confidence"] == "none"
    assert "No APMC mandi" in res["explanation"]


# -----------------------------------------------------------------------------
# 2. Cultivation Cost Calculator Tests
# -----------------------------------------------------------------------------

def test_cacp_state_level_cost(cost_calc):
    """Verify CACP state-level C2 cultivation cost retrieval."""
    res = cost_calc.get_cultivation_cost(state="Punjab", crop="Wheat", year="2018")
    assert res["is_available"] is True
    assert res["cul_cost_c2_per_ha"] is not None
    assert res["cul_cost_c2_per_ha"] > 10000
    assert res["cost_coverage"] == "state_level"
    assert res["is_state_level"] is True
    assert "CACP" in res["explanation"]


def test_cacp_national_benchmark_fallback(cost_calc):
    """Verify national benchmark fallback when a state is not surveyed for that crop."""
    res = cost_calc.get_cultivation_cost(state="Assam", crop="Wheat")
    assert res["is_available"] is True
    assert res["cul_cost_c2_per_ha"] is not None
    assert res["cost_coverage"] == "national_benchmark"
    assert res["is_national_benchmark"] is True


def test_cacp_unavailable_cost(cost_calc):
    """Verify that un-surveyed crops report cost unavailable without fabrication."""
    res = cost_calc.get_cultivation_cost(state="Kerala", crop="Rubber")
    assert res["is_available"] is False
    assert res["cul_cost_c2_per_ha"] is None
    assert res["cost_coverage"] == "cost_unavailable"


# -----------------------------------------------------------------------------
# 3. Financial Calculator Tests
# -----------------------------------------------------------------------------

def test_financial_calculator_complete():
    """Verify exact financial formulas for complete price and cost inputs."""
    # 5 hectares, 2.0 t/ha yield -> 10 tonnes = 100 quintals
    # Modal price = ₹2,000/q -> Gross Revenue = ₹200,000
    # Cultivation cost = ₹30,000/ha -> Total Cost = ₹150,000
    # Profit = ₹50,000 | ROI = (50,000 / 150,000) * 100 = 33.33%
    res = FinancialCalculator.calculate(
        predicted_yield_t_ha=2.0,
        cultivated_area_ha=5.0,
        mandi_price_per_quintal=2000.0,
        cul_cost_c2_per_ha=30000.0,
        crop_name="Wheat",
        state_name="Punjab"
    )

    assert res["total_production_tonnes"] == 10.0
    assert res["total_production_quintals"] == 100.0
    assert res["gross_revenue"] == 200000.0
    assert res["total_cultivation_cost"] == 150000.0
    assert res["estimated_profit"] == 50000.0
    assert res["roi_percent"] == 33.33
    assert res["financial_status"] == "complete"


def test_financial_calculator_missing_cost():
    """Verify that missing cost leaves profit and ROI as None without fabricating zero cost."""
    res = FinancialCalculator.calculate(
        predicted_yield_t_ha=2.0,
        cultivated_area_ha=5.0,
        mandi_price_per_quintal=2000.0,
        cul_cost_c2_per_ha=None
    )
    assert res["gross_revenue"] == 200000.0
    assert res["total_cultivation_cost"] is None
    assert res["estimated_profit"] is None
    assert res["roi_percent"] is None
    assert res["financial_status"] == "cost_unavailable"


def test_financial_calculator_missing_price():
    """Verify that missing price leaves revenue and profit as None without fabricating zero revenue."""
    res = FinancialCalculator.calculate(
        predicted_yield_t_ha=2.0,
        cultivated_area_ha=5.0,
        mandi_price_per_quintal=None,
        cul_cost_c2_per_ha=30000.0
    )
    assert res["gross_revenue"] is None
    assert res["total_cultivation_cost"] == 150000.0
    assert res["estimated_profit"] is None
    assert res["roi_percent"] is None
    assert res["financial_status"] == "price_unavailable"


# -----------------------------------------------------------------------------
# 4. Weather Risk Assessment Tests
# -----------------------------------------------------------------------------

def test_weather_risk_assessment(weather_intel):
    """Verify rainfall lookup and moisture risk evaluation."""
    res = weather_intel.assess_weather_risk(
        state="Punjab",
        district="LUDHIANA",
        season="Kharif",
        crop="Rice"
    )
    assert res["is_observed"] is True
    assert res["mean_seasonal_rainfall_mm"] is not None
    assert res["risk_level"] in ["Low", "Moderate", "High"]
    assert res["water_requirement_category"] == "high"


# -----------------------------------------------------------------------------
# 5. Crop Recommendation Engine Tests
# -----------------------------------------------------------------------------

def test_crop_recommendation_ranking(recommendation_engine):
    """Verify multi-criteria crop ranking and comparison table generation."""
    res = recommendation_engine.recommend_crops(
        state="Punjab",
        district="Ludhiana",
        season="Rabi",
        area_ha=5.0,
        top_n=5
    )

    assert res["state"] == "Punjab"
    assert res["district"] == "Ludhiana"
    assert res["season"] == "Rabi"
    assert res["total_evaluated"] >= 1
    assert len(res["recommendations"]) >= 1

    top_crop = res["recommendations"][0]
    assert top_crop["rank"] == 1
    assert top_crop["predicted_yield_t_ha"] > 0
    assert top_crop["composite_score"] > 0
    assert "Rank" in res["comparison_table"]
    assert "Estimated profit" in top_crop["rationale"] or "Estimated gross revenue" in top_crop["rationale"]


def test_crop_recommendation_unobserved_district(recommendation_engine):
    """Verify safe handling when state/district combination has no historical crops."""
    res = recommendation_engine.recommend_crops(
        state="NonExistentState",
        district="NonExistentDistrict",
        season="Kharif",
        area_ha=5.0
    )
    assert res["total_evaluated"] == 0
    assert len(res["recommendations"]) == 0
    assert "No historical crop records found" in res["comparison_table"]


# -----------------------------------------------------------------------------
# 5. Mandi Market Intelligence & State-District Mapping Tests
# -----------------------------------------------------------------------------

def test_mandi_district_markets_kurnool(mandi_intel):
    """Verify individual APMC markets are retrieved for Kurnool district."""
    res = mandi_intel.get_district_markets(state="Andhra Pradesh", district="Kurnool", crop="Gram")
    assert res["found"] is True
    assert res["district"] == "Kurnool"
    assert len(res["markets"]) > 0
    first = res["markets"][0]
    assert "market" in first
    assert first["modal_price"] is not None
    assert first["modal_price"] > 0
    assert "note" in res


def test_mandi_no_cross_district_leakage(mandi_intel):
    """Verify markets from Punjab do not appear when querying Andhra Pradesh."""
    res = mandi_intel.get_district_markets(state="Andhra Pradesh", district="Kurnool", crop="Gram")
    assert res["found"] is True
    for m in res["markets"]:
        assert "Ludhiana" not in m["market"]
        assert "Jagraon" not in m["market"]
        assert "Khanna" not in m["market"]


def test_cacp_survey_year_displayed(cost_calc):
    """Verify CACP cultivation cost retrieval includes benchmark year and coverage."""
    res = cost_calc.get_cultivation_cost(state="Punjab", crop="Wheat", year="2018")
    assert res["cul_cost_c2_per_ha"] is not None
    assert res["cost_coverage"] in ["state_level", "national_benchmark"]
    assert "benchmark_year" in res
    assert res["benchmark_year"] is not None
    assert "disclaimer" in res


def test_authoritative_state_district_map():
    """Verify state_district_map module separates AP and Telangana correctly."""
    from src.models.state_district_map import get_districts, validate_state_district, is_historical_ap_district

    ap_districts = get_districts("Andhra Pradesh")
    assert len(ap_districts) == 13
    assert "KURNOOL" in ap_districts
    assert "ANANTAPUR" in ap_districts
    assert "WARANGAL" not in ap_districts

    tg_districts = get_districts("Telangana")
    assert len(tg_districts) == 35
    assert "WARANGAL" in tg_districts or "WARANGAL URBAN" in tg_districts
    assert "HYDERABAD" in tg_districts
    assert "ANANTAPUR" not in tg_districts

    assert validate_state_district("Andhra Pradesh", "Kurnool") is True
    assert validate_state_district("Andhra Pradesh", "Warangal") is False
    assert validate_state_district("Telangana", "Warangal") is True

    # Historical helper recognizes undivided AP districts
    assert is_historical_ap_district("Kurnool") is True
    assert is_historical_ap_district("Warangal") is True
    assert is_historical_ap_district("NonExistentDistrict") is False


# -----------------------------------------------------------------------------
# 6. Financial Calculation Audit Automated Tests
# -----------------------------------------------------------------------------

def test_cacp_chronological_latest_survey_year(cost_calc):
    """
    Audit Check: When year is unobserved or a future year is queried,
    CACP cost calculator MUST return the chronologically latest survey year (2021-22),
    not an arbitrary row from the bottom of the CSV file.
    """
    # Test Punjab Wheat
    res_punjab = cost_calc.get_cultivation_cost(state="Punjab", crop="Wheat", year="2026")
    assert res_punjab["is_available"] is True
    assert res_punjab["benchmark_year"] == "2021-22", f"Expected 2021-22, got {res_punjab['benchmark_year']}"
    assert res_punjab["cul_cost_c2_per_ha"] == 69452.42
    assert res_punjab["prod_cost_c2_per_quintal"] == 1500.45

    # Test Andhra Pradesh Gram
    res_ap = cost_calc.get_cultivation_cost(state="Andhra Pradesh", crop="Gram", year="2026")
    assert res_ap["is_available"] is True
    assert res_ap["benchmark_year"] == "2021-22", f"Expected 2021-22, got {res_ap['benchmark_year']}"
    assert res_ap["cul_cost_c2_per_ha"] == 66887.06
    assert res_ap["prod_cost_c2_per_quintal"] == 4105.42


def test_unit_conversion_physical_integrity():
    """
    Audit Check: Unit conversion integrity.
    1 Metric Tonne = exactly 10 Quintals.
    Production in Quintals must equal Production in Tonnes * 10.0.
    """
    res = FinancialCalculator.calculate(
        predicted_yield_t_ha=1.55,
        cultivated_area_ha=5.0,
        mandi_price_per_quintal=2264.02,
        cul_cost_c2_per_ha=69452.42,
        crop_name="Wheat",
        state_name="Punjab"
    )
    # 1.55 t/ha * 5.0 ha = 7.75 tonnes
    assert res["total_production_tonnes"] == 7.75
    # 7.75 tonnes * 10 = 77.50 quintals
    assert res["total_production_quintals"] == 77.50
    assert abs(res["total_production_quintals"] - (res["total_production_tonnes"] * 10.0)) < 1e-6

    # Gross revenue must strictly equal total_production_quintals * mandi_price_per_quintal
    expected_revenue = round(77.50 * 2264.02, 2)
    assert res["gross_revenue"] == expected_revenue

    # Total cost must strictly equal cultivated_area_ha * cul_cost_c2_per_ha
    expected_cost = round(5.0 * 69452.42, 2)
    assert res["total_cultivation_cost"] == expected_cost

    # Profit = Revenue - Cost
    expected_profit = round(expected_revenue - expected_cost, 2)
    assert res["estimated_profit"] == expected_profit

    # ROI = (Profit / Cost) * 100
    expected_roi = round((expected_profit / expected_cost) * 100.0, 2)
    assert res["roi_percent"] == expected_roi


def test_real_data_calculation_punjab_wheat(mandi_intel, cost_calc):
    """
    Audit Real-Data Example 1: Punjab / Ludhiana / Wheat (Rabi, 5.0 ha).
    District-level APMC market match + state-level CACP survey.
    """
    price_info = mandi_intel.get_price_intelligence("Punjab", "Ludhiana", "Wheat")
    assert price_info["price_level"] == "district"
    assert price_info["price_per_quintal"] == 2264.02
    assert price_info["latest_year"] == 2026
    assert price_info["markets_covered"] > 0

    cost_info = cost_calc.get_cultivation_cost("Punjab", "Wheat")
    assert cost_info["cost_coverage"] == "state_level"
    assert cost_info["benchmark_year"] == "2021-22"
    assert cost_info["cul_cost_c2_per_ha"] == 69452.42

    fin = FinancialCalculator.calculate(
        predicted_yield_t_ha=1.55,
        cultivated_area_ha=5.0,
        mandi_price_per_quintal=price_info["price_per_quintal"],
        cul_cost_c2_per_ha=cost_info["cul_cost_c2_per_ha"],
        crop_name="Wheat",
        state_name="Punjab",
        price_level=price_info["price_level"],
        cost_level=cost_info["cost_coverage"]
    )

    assert fin["financial_status"] == "complete"
    assert fin["total_production_tonnes"] == 7.75
    assert fin["total_production_quintals"] == 77.50
    assert fin["gross_revenue"] == 175461.55
    assert fin["total_cultivation_cost"] == 347262.10
    assert fin["estimated_profit"] == -171800.55
    assert fin["roi_percent"] == -49.47


def test_real_data_calculation_ap_gram(mandi_intel, cost_calc):
    """
    Audit Real-Data Example 2: Andhra Pradesh / Kurnool / Gram (Rabi, 3.0 ha).
    District-level APMC market match + state-level CACP survey.
    """
    price_info = mandi_intel.get_price_intelligence("Andhra Pradesh", "Kurnool", "Gram")
    assert price_info["price_level"] == "district"
    assert price_info["price_per_quintal"] == 5129.56
    assert price_info["latest_year"] == 2026

    cost_info = cost_calc.get_cultivation_cost("Andhra Pradesh", "Gram")
    assert cost_info["cost_coverage"] == "state_level"
    assert cost_info["benchmark_year"] == "2021-22"
    assert cost_info["cul_cost_c2_per_ha"] == 66887.06

    fin = FinancialCalculator.calculate(
        predicted_yield_t_ha=1.01,
        cultivated_area_ha=3.0,
        mandi_price_per_quintal=price_info["price_per_quintal"],
        cul_cost_c2_per_ha=cost_info["cul_cost_c2_per_ha"],
        crop_name="Gram",
        state_name="Andhra Pradesh",
        price_level=price_info["price_level"],
        cost_level=cost_info["cost_coverage"]
    )

    assert fin["financial_status"] == "complete"
    assert fin["total_production_tonnes"] == 3.03
    assert fin["total_production_quintals"] == 30.30
    assert fin["gross_revenue"] == 155425.67
    assert fin["total_cultivation_cost"] == 200661.18
    assert fin["estimated_profit"] == -45235.51
    assert fin["roi_percent"] == -22.54


def test_real_data_calculation_missing_cacp_cost(mandi_intel, cost_calc):
    """
    Audit Real-Data Example 3A: Crop with observed APMC mandi prices but unobserved CACP cost.
    Ensures gross revenue is calculated but cost, profit, and ROI remain None (cost_unavailable).
    """
    price_info = mandi_intel.get_price_intelligence("Karnataka", "Bangalore", "Coriander")
    assert price_info["price_per_quintal"] is not None
    assert price_info["price_per_quintal"] > 0

    cost_info = cost_calc.get_cultivation_cost("Karnataka", "Coriander")
    assert cost_info["is_available"] is False
    assert cost_info["cul_cost_c2_per_ha"] is None
    assert cost_info["cost_coverage"] == "cost_unavailable"

    fin = FinancialCalculator.calculate(
        predicted_yield_t_ha=1.5,
        cultivated_area_ha=2.0,
        mandi_price_per_quintal=price_info["price_per_quintal"],
        cul_cost_c2_per_ha=cost_info["cul_cost_c2_per_ha"],
        crop_name="Coriander",
        state_name="Karnataka",
        price_level=price_info["price_level"],
        cost_level=cost_info["cost_coverage"]
    )

    assert fin["financial_status"] == "cost_unavailable"
    assert fin["gross_revenue"] is not None
    assert fin["gross_revenue"] > 0
    assert fin["total_cultivation_cost"] is None
    assert fin["estimated_profit"] is None
    assert fin["roi_percent"] is None
    assert "CACP cultivation cost unobserved" in fin["explanation"]


def test_real_data_calculation_missing_mandi_price(mandi_intel, cost_calc):
    """
    Audit Real-Data Example 3B: Crop with unobserved APMC mandi price.
    Ensures cultivation cost is calculated but revenue, profit, and ROI remain None (price_unavailable).
    """
    # Simulated missing price scenario
    cost_info = cost_calc.get_cultivation_cost("Punjab", "Wheat")
    assert cost_info["is_available"] is True

    fin = FinancialCalculator.calculate(
        predicted_yield_t_ha=2.0,
        cultivated_area_ha=4.0,
        mandi_price_per_quintal=None,
        cul_cost_c2_per_ha=cost_info["cul_cost_c2_per_ha"],
        crop_name="Wheat",
        state_name="Punjab",
        price_level="unavailable",
        cost_level=cost_info["cost_coverage"]
    )

    assert fin["financial_status"] == "price_unavailable"
    assert fin["gross_revenue"] is None
    assert fin["total_cultivation_cost"] == round(4.0 * cost_info["cul_cost_c2_per_ha"], 2)
    assert fin["estimated_profit"] is None
    assert fin["roi_percent"] is None
    assert "Mandi price unobserved" in fin["explanation"]


def test_zero_and_negative_area_safety():
    """
    Audit Check: Area = 0.0 or negative must not cause division-by-zero or crashes.
    """
    res_zero = FinancialCalculator.calculate(
        predicted_yield_t_ha=2.0,
        cultivated_area_ha=0.0,
        mandi_price_per_quintal=2000.0,
        cul_cost_c2_per_ha=30000.0
    )
    assert res_zero["cultivated_area_ha"] == 0.0
    assert res_zero["total_production_tonnes"] == 0.0
    assert res_zero["total_production_quintals"] == 0.0
    assert res_zero["gross_revenue"] == 0.0
    assert res_zero["total_cultivation_cost"] == 0.0
    assert res_zero["estimated_profit"] == 0.0
    assert res_zero["roi_percent"] == 0.0

    res_neg = FinancialCalculator.calculate(
        predicted_yield_t_ha=2.0,
        cultivated_area_ha=-5.0,
        mandi_price_per_quintal=2000.0,
        cul_cost_c2_per_ha=30000.0
    )
    assert res_neg["cultivated_area_ha"] == 0.0
    assert res_neg["total_production_tonnes"] == 0.0



