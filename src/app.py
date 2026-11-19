import os
import json
from pathlib import Path
import pandas as pd
from flask import Flask, render_template

from src.routes.predict import predict_bp, predictor, mandi_intel, cost_calc, weather_intel
from src.models.state_district_map import get_districts as get_auth_districts
from src.models.year_utils import FUTURE_AGRICULTURAL_YEARS, parse_agricultural_year

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

app.register_blueprint(predict_bp)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OPTIONS_PATH = MODELS_DIR / "dataset_options.json"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
COMPARISON_CSV = MODELS_DIR / "model_comparison.csv"

# Project identity constants
PROJECT_TITLE = "Machine Learning Based Crop Yield Prediction & Recommendation System"
PROJECT_SHORT = "ML Crop Yield Prediction"


@app.context_processor
def inject_dataset_options():
    """Inject dropdown selections and state-district hierarchies into templates."""
    data = predictor.data

    states = sorted(data["State"].unique().tolist())
    districts = sorted(data["District"].unique().tolist())
    crops = sorted(data["Crop"].unique().tolist())
    seasons = sorted(data["Season"].unique().tolist())

    # Future agricultural years for farmer prediction & planning forms (e.g. '2026-27', '2027-28')
    future_years = list(FUTURE_AGRICULTURAL_YEARS)
    # Historical years preserved internally for dataset validation and reference
    historical_years = sorted(data["Year"].unique().tolist(), reverse=True)

    # Build state -> districts mapping with authoritative AP/TG correction
    state_districts = {}
    raw_opts = {}
    if OPTIONS_PATH.exists():
        try:
            with open(OPTIONS_PATH, "r") as f:
                opts = json.load(f)
                raw_opts = opts.get("districts", {})
        except Exception:
            pass

    if not raw_opts:
        for st in states:
            raw_opts[st] = sorted(data[data["State"] == st]["District"].unique().tolist())

    # Apply authoritative override for AP (post-bifurcation) and Telangana
    for st in states:
        raw_list = raw_opts.get(st, [])
        state_districts[st] = get_auth_districts(st, fallback_districts=raw_list)

    return {
        "states": states,
        "districts": districts,
        "crops": crops,
        "years": future_years,              # Main 'years' variable now strictly provides future years
        "future_years": future_years,       # Explicit future agricultural years
        "historical_years": historical_years, # Historical years retained internally
        "seasons": seasons,
        "state_districts_json": json.dumps(state_districts),
        "project_title": PROJECT_TITLE,
        "project_short": PROJECT_SHORT,
    }


@app.context_processor
def inject_project_title():
    """Make project title available to all templates."""
    return {
        "project_title": PROJECT_TITLE,
        "project_short": PROJECT_SHORT,
    }


@app.route("/")
def home():
    """Home / Predictor page."""
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    """System Dashboard: Data coverage, model scores, and recent mandi price benchmarks."""
    metrics_data = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                metrics_data = json.load(f)
        except Exception:
            pass

    # Recent Mandi Price Benchmarks across primary agricultural commodities
    key_crops = [
        "Wheat", "Rice", "Maize", "Cotton(lint)", "Rapeseed &Mustard",
        "Groundnut", "Soyabean", "Gram", "Potato", "Onion"
    ]
    price_samples = []
    if mandi_intel.df_national is not None:
        for crop in key_crops:
            norm = mandi_intel.normalize_crop_name(crop)
            sub = mandi_intel.df_national[mandi_intel.df_national["Crop_Upper"] == norm.upper()]
            if len(sub) > 0:
                row = sub.iloc[0]
                price_samples.append({
                    "crop": crop,
                    "mandi_commodity": norm,
                    "recent_modal_price": float(row["recent_avg_modal_price"]) if pd.notna(row["recent_avg_modal_price"]) else None,
                    "historical_avg_price": float(row["historical_avg_modal_price"]) if pd.notna(row["historical_avg_modal_price"]) else None,
                    "recent_records": int(row["recent_records"]),
                    "latest_year": int(row["latest_year"]) if pd.notna(row["latest_year"]) else None
                })

    return render_template(
        "dashboard.html",
        metrics_data=metrics_data,
        price_samples=price_samples
    )


@app.route("/analytics")
def analytics():
    """Analytics & Model Evaluation: Comparison of all 5 algorithms, feature importance, and diagnostic charts."""
    comparison_rows = []
    if COMPARISON_CSV.exists():
        try:
            df_comp = pd.read_csv(COMPARISON_CSV)
            comparison_rows = df_comp.to_dict(orient="records")
        except Exception:
            pass

    metrics_data = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                metrics_data = json.load(f)
        except Exception:
            pass

    plots = [
        {"id": "prod_comp", "title": "Production Models Comparison (R², RMSE, MAE)", "filename": "plots/production_prediction_model_comparison.png"},
        {"id": "prod_fit", "title": "Production Model: Actual vs Predicted", "filename": "plots/production_prediction_actual_vs_predicted.png"},
        {"id": "prod_imp", "title": "Production Model: Feature Importance", "filename": "plots/production_prediction_feature_importance.png"},
        {"id": "prod_res", "title": "Production Model: Residual Distribution", "filename": "plots/production_prediction_residuals.png"},
        {"id": "yield_comp", "title": "Yield Models Comparison (R², RMSE, MAE)", "filename": "plots/yield_prediction_model_comparison.png"},
        {"id": "yield_fit", "title": "Yield Model: Actual vs Predicted", "filename": "plots/yield_prediction_actual_vs_predicted.png"},
        {"id": "yield_imp", "title": "Yield Model: Feature Importance", "filename": "plots/yield_prediction_feature_importance.png"},
        {"id": "yield_res", "title": "Yield Model: Residual Distribution", "filename": "plots/yield_prediction_residuals.png"},
    ]

    return render_template(
        "analytics.html",
        comparison_rows=comparison_rows,
        metrics_data=metrics_data,
        plots=plots
    )


if __name__ == "__main__":
    app.run(debug=True)
