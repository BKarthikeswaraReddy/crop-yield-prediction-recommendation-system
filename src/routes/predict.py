import os
import json
import math
from pathlib import Path
from flask import Blueprint, request, render_template, jsonify

from src.models.predictor import Predictor
from src.models.mandi_intelligence import MandiIntelligence
from src.models.cost_calculator import CultivationCostCalculator
from src.models.financial_calculator import FinancialCalculator
from src.models.weather_intelligence import WeatherIntelligence
from src.models.decision_engine import CropRecommendationEngine
from src.models.state_district_map import get_districts as get_auth_districts, validate_state_district
from src.models.year_utils import parse_agricultural_year, FUTURE_AGRICULTURAL_YEARS, is_valid_future_year

predict_bp = Blueprint('predict', __name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = DATA_DIR / "India Agriculture Crop Production.csv"
if not DATA_PATH.exists() and (DATA_DIR / "raw" / "India Agriculture Crop Production.csv").exists():
    DATA_PATH = DATA_DIR / "raw" / "India Agriculture Crop Production.csv"

# Initialize singletons
predictor = Predictor(str(DATA_PATH))
mandi_intel = MandiIntelligence(DATA_DIR)
cost_calc = CultivationCostCalculator(DATA_DIR)
weather_intel = WeatherIntelligence(DATA_DIR / "weather")
recommendation_engine = CropRecommendationEngine(
    predictor=predictor,
    mandi_intel=mandi_intel,
    cost_calc=cost_calc,
    weather_intel=weather_intel
)

# Honest model horizon disclaimer shown with every prediction result
YIELD_DISCLAIMER = (
    "This yield estimate is based on historical agricultural patterns for this region and crop. "
    "The trained model evaluates crop type, district, season, and cultivated area. "
    "Predictions do not vary across future agricultural years because year is not a split-differentiating feature in the validated model. "
    "A time-series autoregressive forecasting model is not serialized in this release. "
    "Use these predictions as a data-driven planning benchmark for future planning purposes."
)


@predict_bp.route('/predict', methods=['POST'])
def predict():
    """Predict crop yield, total production, and complete farm economics."""
    state = request.form.get('state')
    district = request.form.get('district')
    crop = request.form.get('crop')
    year = request.form.get('year')
    season = request.form.get('season')
    area = request.form.get('area')

    if not all([state, district, crop, year, season, area]):
        return render_template(
            'index.html',
            error='Please fill in all required fields.'
        )

    try:
        area_float = float(area)
        if not math.isfinite(area_float) or area_float <= 0:
            raise ValueError('Area must be greater than zero.')

        if predictor.model is None:
            predictor.train_model()

        # Parse agricultural year (e.g. '2026-27' -> start 2026, end 2027)
        year_info = parse_agricultural_year(year)

        # 1. Physical ML Predictions
        # Use validated yield model (t/ha) to derive farm-scale production:
        # farm_production_tonnes = predicted_yield * cultivated_area_ha
        # (production_model.joblib is trained on district-level census aggregates
        # and is NOT suitable for individual farmer plot sizes.)
        predicted_yield = predictor.predict_yield(
            state=state,
            district=district,
            crop=crop,
            year=year_info["formatted"],
            season=season,
            area=area_float
        )
        farm_production_tonnes = round(predicted_yield * area_float, 2)

        # 2. Mandi Price Intelligence
        price_info = mandi_intel.get_price_intelligence(state, district, crop)

        # 3. Cultivation Cost Calculator
        cost_info = cost_calc.get_cultivation_cost(state, crop, str(year_info["start_year"]))

        # 4. Financial Economics (uses exact same farm_production_tonnes = yield * area)
        financials = FinancialCalculator.calculate(
            predicted_yield_t_ha=predicted_yield,
            cultivated_area_ha=area_float,
            mandi_price_per_quintal=price_info.get("price_per_quintal"),
            cul_cost_c2_per_ha=cost_info.get("cul_cost_c2_per_ha"),
            crop_name=crop,
            state_name=state,
            price_level=price_info.get("price_level", "unavailable"),
            cost_level=cost_info.get("cost_coverage", "unavailable")
        )

        # 5. Weather Risk Assessment
        weather_info = weather_intel.assess_weather_risk(state, district, season, crop)

        # 6. Individual APMC Market Records & Per-Market Financial Scenarios
        district_markets = mandi_intel.get_district_markets(state, district, crop)
        market_scenarios = []
        if district_markets and district_markets.get("found"):
            market_scenarios = FinancialCalculator.calculate_market_scenarios(
                predicted_yield_t_ha=predicted_yield,
                cultivated_area_ha=area_float,
                markets=district_markets.get("markets", []),
                cul_cost_c2_per_ha=cost_info.get("cul_cost_c2_per_ha"),
                crop_name=crop,
                state_name=state,
                cost_level=cost_info.get("cost_coverage", "unavailable"),
            )

        return render_template(
            'index.html',
            prediction=financials["total_production_tonnes"],
            farm_production_tonnes=farm_production_tonnes,
            predicted_yield=round(predicted_yield, 2),
            selected_state=state,
            selected_district=district,
            selected_crop=crop,
            selected_year=year_info["formatted"],
            year_info=year_info,
            selected_season=season,
            selected_area=area_float,
            metrics=predictor.metrics,
            financials=financials,
            price_info=price_info,
            cost_info=cost_info,
            weather_info=weather_info,
            district_markets=district_markets,
            market_scenarios=market_scenarios,
            yield_disclaimer=YIELD_DISCLAIMER
        )

    except ValueError as error:
        return render_template(
            'index.html',
            error=str(error)
        )

    except Exception as error:
        return render_template(
            'index.html',
            error=f'Prediction failed: {error}'
        )


@predict_bp.route('/recommend', methods=['GET', 'POST'])
def recommend():
    """Farmer Decision Support: multi-criteria crop recommendation for selected district and season."""
    recommendation_results = None
    error = None

    selected_state = request.form.get('state') if request.method == 'POST' else request.args.get('state')
    selected_district = request.form.get('district') if request.method == 'POST' else request.args.get('district')
    selected_season = request.form.get('season') if request.method == 'POST' else request.args.get('season')
    selected_area = request.form.get('area') if request.method == 'POST' else request.args.get('area')
    raw_year = request.form.get('year') if request.method == 'POST' else request.args.get('year', FUTURE_AGRICULTURAL_YEARS[0])
    year_info = parse_agricultural_year(raw_year)
    selected_year = year_info["formatted"]

    if request.method == 'POST' or (selected_state and selected_district and selected_season and selected_area):
        if not all([selected_state, selected_district, selected_season, selected_area]):
            error = "Please provide State, District, Season, and Cultivated Area to generate recommendations."
        else:
            try:
                area_float = float(selected_area)
                if area_float <= 0:
                    raise ValueError("Cultivated area must be greater than zero.")

                recommendation_results = recommendation_engine.recommend_crops(
                    state=selected_state,
                    district=selected_district,
                    season=selected_season,
                    area_ha=area_float,
                    year=selected_year,
                    top_n=6
                )
            except ValueError as ve:
                error = str(ve)
            except Exception as e:
                error = f"Recommendation analysis failed: {e}"

    return render_template(
        'recommend.html',
        selected_state=selected_state,
        selected_district=selected_district,
        selected_season=selected_season,
        selected_area=selected_area,
        selected_year=selected_year,
        year_info=year_info,
        results=recommendation_results,
        yield_disclaimer=YIELD_DISCLAIMER,
        error=error
    )


@predict_bp.route('/weather', methods=['GET', 'POST'])
def weather():
    """Weather Intelligence: Seasonal rainfall analysis, moisture risk, and irrigation advisories."""
    selected_state = request.form.get('state') if request.method == 'POST' else request.args.get('state')
    selected_district = request.form.get('district') if request.method == 'POST' else request.args.get('district')
    selected_season = request.form.get('season') if request.method == 'POST' else request.args.get('season', 'Kharif')
    selected_crop = request.form.get('crop') if request.method == 'POST' else request.args.get('crop')

    weather_data = None
    temp_data = None
    district_seasonal_history = []
    error = None

    if selected_district:
        try:
            # Query seasonal rainfall table (state-disambiguated)
            district_seasonal_history = weather_intel.get_district_seasonal_history(
                state=selected_state,
                district=selected_district,
            )

            eval_crop = selected_crop if selected_crop else ("Wheat" if "rabi" in selected_season.lower() else "Rice")
            weather_data = weather_intel.assess_weather_risk(
                state=selected_state or "",
                district=selected_district,
                season=selected_season,
                crop=eval_crop
            )
            temp_data = weather_intel.get_state_temperature_profile(
                state=selected_state,
                season=selected_season,
                crop=eval_crop,
            )
        except Exception as e:
            error = f"Weather lookup failed: {e}"

    return render_template(
        'weather.html',
        selected_state=selected_state,
        selected_district=selected_district,
        selected_season=selected_season,
        selected_crop=selected_crop,
        weather_data=weather_data,
        temp_data=temp_data,
        seasonal_history=district_seasonal_history,
        water_requirements=weather_intel.crop_water_requirements,
        error=error
    )


@predict_bp.route('/api/districts', methods=['GET'])
def get_districts_api():
    """API endpoint returning valid districts for the selected state.

    Uses the authoritative post-bifurcation mapping for AP and Telangana;
    falls back to raw dataset districts for all other states.
    """
    state = request.args.get('state')
    if not state or predictor.data is None:
        return jsonify([])

    # Get raw districts from dataset as fallback
    sub = predictor.data[predictor.data["State"].str.upper() == str(state).strip().upper()]
    raw_districts = sorted(sub["District"].unique().tolist())

    # Apply authoritative override for AP/TG bifurcation correction
    districts = get_auth_districts(state, fallback_districts=raw_districts)
    return jsonify(districts)


@predict_bp.route('/api/price', methods=['GET'])
def get_price():
    """API endpoint returning APMC modal price intelligence."""
    state = request.args.get('state', '')
    district = request.args.get('district', '')
    crop = request.args.get('crop', '')

    if not crop:
        return jsonify({"error": "Crop parameter required"}), 400

    info = mandi_intel.get_price_intelligence(state, district, crop)
    return jsonify(info)


@predict_bp.route('/api/mandi_markets', methods=['GET'])
def get_mandi_markets():
    """API endpoint returning individual APMC market records for a district and crop.

    Query params:
        state    – state name
        district – district name
        crop     – production crop name (e.g. "Wheat", "Rice")
        limit    – max number of markets to return (default 20, max 50)

    Returns JSON with:
        found    – bool
        markets  – list of individual market records with Commodity, Variety,
                   Grade, Min_Price, Max_Price, Modal_Price, Arrival_Date
        note     – mandatory disclaimer
    """
    state = request.args.get('state', '')
    district = request.args.get('district', '')
    crop = request.args.get('crop', '')

    try:
        limit = min(int(request.args.get('limit', 20)), 50)
    except (ValueError, TypeError):
        limit = 20

    if not crop:
        return jsonify({"error": "crop parameter required"}), 400
    if not district:
        return jsonify({"error": "district parameter required"}), 400

    result = mandi_intel.get_district_markets(state, district, crop, max_markets=limit)
    return jsonify(result)
