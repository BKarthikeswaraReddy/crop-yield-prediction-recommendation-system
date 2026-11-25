import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = ROOT_DIR

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

LEGACY_DIR = os.path.join(ROOT_DIR, "crop_Production_prediction_new", "crop-production-predictor")
if os.path.exists(LEGACY_DIR) and LEGACY_DIR not in sys.path:
    sys.path.insert(1, LEGACY_DIR)

import pytest
from src.app import app
from src.routes.predict import predictor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_predictor_data_loaded():
    """Verify that dataset loaded properly with usable records."""
    assert predictor.data is not None
    assert len(predictor.data) > 300000
    for col in ['State', 'District', 'Crop', 'Year', 'Season', 'Area', 'Production']:
        assert col in predictor.data.columns


def test_home_route(client):
    """Verify GET / returns 200 and contains official branding and form elements."""
    response = client.get("/")
    assert response.status_code == 200
    content = response.data.decode("utf-8")
    assert "Machine Learning Based Crop Yield Prediction" in content
    assert "Cultivated Area" in content


def test_dashboard_route(client):
    """Verify GET /dashboard returns 200 and contains overview details and official branding."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    content = response.data.decode("utf-8")
    assert "Machine Learning Based Crop Yield Prediction" in content
    assert "Crop Production Dashboard" in content
    assert "326,999" in content


def test_analytics_route(client):
    """Verify GET /analytics returns 200 and contains analytics architecture and official branding."""
    response = client.get("/analytics")
    assert response.status_code == 200
    content = response.data.decode("utf-8")
    assert "Machine Learning Based Crop Yield Prediction" in content
    assert "Agricultural Production Analytics" in content


def test_predict_missing_fields(client):
    """Verify POST /predict with missing fields displays validation error."""
    response = client.post("/predict", data={
        "state": "Assam",
        "district": "BAKSA",
        "crop": "Rice",
        "year": "2014",
        "season": "Kharif"
        # 'area' is intentionally omitted
    })
    assert response.status_code == 200
    content = response.data.decode("utf-8")
    assert "Please fill in all required fields." in content


def test_predict_invalid_area(client):
    """Verify POST /predict with negative or zero area shows error."""
    response = client.post("/predict", data={
        "state": "Assam",
        "district": "BAKSA",
        "crop": "Rice",
        "year": "2014",
        "season": "Kharif",
        "area": "-10"
    })
    assert response.status_code == 200
    content = response.data.decode("utf-8")
    assert "Area must be greater than zero." in content


def test_predictor_predict_calculation():
    """Verify Predictor.predict returns valid non-negative float using a fitted model."""
    original_model = predictor.model
    try:
        # Fit a quick dummy regressor on a small sample to test predict method without waiting for 100 RF trees
        sample = predictor.data.head(50)
        cat_features = ['State', 'District', 'Crop', 'Year', 'Season']
        num_features = ['Area']
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
                ('num', 'passthrough', num_features)
            ]
        )
        dummy_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', DummyRegressor(strategy='mean'))
        ])
        dummy_pipeline.fit(sample[cat_features + num_features], sample['Production'])
        predictor.model = dummy_pipeline

        prediction = predictor.predict(
            state="Assam",
            district="BAKSA",
            crop="Rice",
            year="2014",
            season="Kharif",
            area=50.0
        )
        assert isinstance(prediction, float)
        assert prediction >= 0.0
    finally:
        predictor.model = original_model


def test_predict_valid_submission(client):
    """Verify POST /predict with valid fields renders prediction and metrics."""
    original_model = predictor.model
    original_metrics = predictor.metrics
    try:
        sample = predictor.data.head(50)
        cat_features = ['State', 'District', 'Crop', 'Year', 'Season']
        num_features = ['Area']
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
                ('num', 'passthrough', num_features)
            ]
        )
        dummy_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', DummyRegressor(strategy='mean'))
        ])
        dummy_pipeline.fit(sample[cat_features + num_features], sample['Production'])
        predictor.model = dummy_pipeline
        predictor.metrics = {'MAE': 12.34, 'RMSE': 45.67, 'R2 Score': 0.89}

        response = client.post("/predict", data={
            "state": "Assam",
            "district": "BAKSA",
            "crop": "Rice",
            "year": "2014",
            "season": "Kharif",
            "area": "100"
        })
        assert response.status_code == 200
        content = response.data.decode("utf-8")
        assert "PREDICTED PRODUCTION" in content
        assert "Model Evaluation" in content
        assert "MAE" in content
        assert "12.3400" in content
    finally:
        predictor.model = original_model
        predictor.metrics = original_metrics


def test_serialized_models_exist():
    """Verify serialized models, metrics, options, and comparison plots exist on disk."""
    models_dir = os.path.join(PROJECT_DIR, "models")
    plots_dir = os.path.join(models_dir, "plots")

    assert os.path.isfile(os.path.join(models_dir, "production_model.joblib"))
    assert os.path.isfile(os.path.join(models_dir, "yield_model.joblib"))
    assert os.path.isfile(os.path.join(models_dir, "model_metrics.json"))
    assert os.path.isfile(os.path.join(models_dir, "dataset_options.json"))
    assert os.path.isfile(os.path.join(models_dir, "model_comparison.csv"))

    expected_plots = [
        "production_prediction_actual_vs_predicted.png",
        "production_prediction_feature_importance.png",
        "production_prediction_model_comparison.png",
        "production_prediction_residuals.png",
        "yield_prediction_actual_vs_predicted.png",
        "yield_prediction_feature_importance.png",
        "yield_prediction_model_comparison.png",
        "yield_prediction_residuals.png",
    ]
    for plot_name in expected_plots:
        assert os.path.isfile(os.path.join(plots_dir, plot_name)), f"Missing plot: {plot_name}"


def test_production_and_yield_inference():
    """Verify inference with both production and yield serialized models."""
    assert predictor.production_model is not None
    assert predictor.yield_model is not None

    prod = predictor.predict(
        state="Punjab",
        district="LUDHIANA",
        crop="Wheat",
        year="2018",
        season="Rabi",
        area=100.0
    )
    assert isinstance(prod, float)
    assert prod > 0.0

    yield_val = predictor.predict_yield(
        state="Punjab",
        district="LUDHIANA",
        crop="Wheat",
        year="2018",
        season="Rabi",
        area=100.0
    )
    assert isinstance(yield_val, float)
    assert yield_val > 0.0


def test_model_metrics_loaded():
    """Verify metrics dictionary is loaded from model_metrics.json."""
    assert predictor.metrics is not None
    assert "MAE" in predictor.metrics
    assert "RMSE" in predictor.metrics
    assert "R2 Score" in predictor.metrics
    assert predictor.metrics["R2 Score"] > 0.90


def test_recommend_get_route(client):
    """Verify GET /recommend returns 200 with recommendation form."""
    response = client.get("/recommend")
    assert response.status_code == 200
    content = response.data.decode("utf-8")
    assert "Crop Recommendation Engine" in content
    assert "Land Area" in content


def test_recommend_post_route(client):
    """Verify POST /recommend returns 200 with ranked recommendations table."""
    response = client.post("/recommend", data={
        "state": "Punjab",
        "district": "Ludhiana",
        "season": "Rabi",
        "area": "5.0"
    })
    assert response.status_code == 200
    content = response.data.decode("utf-8")
    assert "Recommended Crops for Ludhiana, Punjab" in content
    assert "Top Recommended Crop" in content
    assert "Multi-Crop Comparison" in content


def test_recommend_post_missing_fields(client):
    """Verify POST /recommend with missing fields returns notice."""
    response = client.post("/recommend", data={
        "state": "Punjab",
        "district": "Ludhiana"
        # season and area omitted
    })
    assert response.status_code == 200
    content = response.data.decode("utf-8")
    assert "Please provide State, District, Season, and Cultivated Area" in content


def test_weather_get_route(client):
    """Verify GET /weather returns 200 with climate search form."""
    response = client.get("/weather")
    assert response.status_code == 200
    content = response.data.decode("utf-8")
    assert "District Weather & Rainfall Intelligence" in content


def test_weather_post_route(client):
    """Verify POST /weather returns 200 with moisture risk profile and seasonal breakdown."""
    response = client.post("/weather", data={
        "state": "Punjab",
        "district": "Ludhiana",
        "season": "Kharif"
    })
    assert response.status_code == 200
    content = response.data.decode("utf-8")
    assert "Weather Risk Profile" in content
    assert "Moisture Risk:" in content
    assert "All-Season Rainfall Breakdown" in content


def test_api_districts(client):
    """Verify GET /api/districts returns JSON list of districts for the state."""
    response = client.get("/api/districts?state=Punjab")
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert any("LUDHIANA" in d.upper() for d in data)


def test_api_price(client):
    """Verify GET /api/price returns JSON price intelligence with modal price."""
    response = client.get("/api/price?state=Punjab&district=Ludhiana&crop=Wheat")
    assert response.status_code == 200
    data = response.get_json()
    assert data["crop"] == "Wheat"
    assert data["price_per_quintal"] is not None
    assert data["price_per_quintal"] > 0
    assert data["price_level"] in ["district", "state", "national"]


def test_predict_post_with_financials_and_weather(client):
    """Verify POST /predict returns 200 and renders Farm Financial Analysis and Weather Risk cards."""
    response = client.post("/predict", data={
        "state": "Punjab",
        "district": "Ludhiana",
        "crop": "Wheat",
        "year": "2018",
        "season": "Rabi",
        "area": "5.0"
    })
    assert response.status_code == 200
    content = response.data.decode("utf-8")
    assert "PREDICTED PRODUCTION" in content
    assert "Farmer Financial Analysis" in content
    assert "Expected Mandi Price" in content
    assert "Gross Revenue" in content
    assert "Total Cost" in content
    assert "Seasonal Weather & Moisture Advisory" in content


def test_ap_districts_no_telangana(client):
    """Verify AP districts returned by /api/districts do not contain Telangana districts."""
    response = client.get("/api/districts?state=Andhra Pradesh")
    assert response.status_code == 200
    districts = response.get_json()
    assert isinstance(districts, list)
    assert len(districts) == 13
    assert "KURNOOL" in districts
    assert "ANANTAPUR" in districts
    # Telangana districts must NOT be present under Andhra Pradesh
    assert "WARANGAL" not in districts
    assert "ADILABAD" not in districts
    assert "KARIMNAGAR" not in districts
    assert "HYDERABAD" not in districts


def test_tg_districts_contains_telangana(client):
    """Verify Telangana districts returned by /api/districts contain TG districts."""
    response = client.get("/api/districts?state=Telangana")
    assert response.status_code == 200
    districts = response.get_json()
    assert isinstance(districts, list)
    assert len(districts) == 35
    assert "WARANGAL" in districts or "WARANGAL URBAN" in districts
    assert "ADILABAD" in districts
    assert "HYDERABAD" in districts
    # Core modern AP-only districts must NOT be in Telangana
    assert "ANANTAPUR" not in districts
    assert "CHITTOOR" not in districts


def test_yield_disclaimer_rendered(client):
    """Verify that POST /predict renders the honest planning benchmark disclaimer."""
    response = client.post("/predict", data={
        "state": "Punjab",
        "district": "Ludhiana",
        "crop": "Wheat",
        "year": "2025",
        "season": "Rabi",
        "area": "2.5"
    })
    assert response.status_code == 200
    content = response.data.decode("utf-8")
    assert "Model Note:" in content
    assert "data-driven planning benchmark" in content


def test_api_mandi_markets(client):
    """Verify GET /api/mandi_markets returns individual APMC market records."""
    response = client.get("/api/mandi_markets?state=Punjab&district=Ludhiana&crop=Wheat")
    assert response.status_code == 200
    data = response.get_json()
    assert data["found"] is True
    assert data["crop"] == "Wheat"
    assert len(data["markets"]) > 0
    first_market = data["markets"][0]
    assert "market" in first_market
    assert "modal_price" in first_market
    assert "arrival_date" in first_market
    assert "note" in data


def test_api_mandi_markets_missing_params(client):
    """Verify GET /api/mandi_markets returns 400 when required parameters are missing."""
    resp1 = client.get("/api/mandi_markets?district=Ludhiana")
    assert resp1.status_code == 400

    resp2 = client.get("/api/mandi_markets?crop=Wheat")
    assert resp2.status_code == 400


def test_api_mandi_markets_no_cross_district(client):
    """Verify that mandi markets queried for Ludhiana do not return markets from another district."""
    response = client.get("/api/mandi_markets?state=Punjab&district=Ludhiana&crop=Wheat")
    assert response.status_code == 200
    data = response.get_json()
    assert data["found"] is True
    for m in data["markets"]:
        # Verify no Kurnool / AP markets leaked in
        assert "Nandyal" not in m["market"]
        assert "Adoni" not in m["market"]


def test_future_prediction_year_dropdown_options(client):
    """Verify farmer forms render ONLY future agricultural years (2026-27 onwards), not historical years."""
    import re
    from src.models.year_utils import FUTURE_AGRICULTURAL_YEARS

    # Check Prediction form (GET /)
    resp = client.get("/")
    assert resp.status_code == 200
    content = resp.data.decode("utf-8")
    assert "Future Prediction Year" in content
    assert 'name="year"' in content

    # Find the year select element
    m = re.search(r'<select name="year"[^>]*>(.*?)</select>', content, re.DOTALL)
    assert m is not None
    select_html = m.group(1)

    # Historical years must NOT be selectable options
    for h_year in ["1997", "2000", "2010", "2015", "2018", "2019", "2020"]:
        assert f'value="{h_year}"' not in select_html

    # Future agricultural years MUST be selectable options
    for f_year in FUTURE_AGRICULTURAL_YEARS:
        assert f'value="{f_year}"' in select_html

    # Check Recommendation form (GET /recommend)
    resp_rec = client.get("/recommend")
    assert resp_rec.status_code == 200
    content_rec = resp_rec.data.decode("utf-8")
    assert "Future Prediction Year" in content_rec
    m_rec = re.search(r'<select name="year"[^>]*>(.*?)</select>', content_rec, re.DOTALL)
    assert m_rec is not None
    for f_year in FUTURE_AGRICULTURAL_YEARS:
        assert f'value="{f_year}"' in m_rec.group(1)


def test_agricultural_year_parsing():
    """Verify parse_agricultural_year decomposes standard agricultural years correctly."""
    from src.models.year_utils import parse_agricultural_year

    # Test standard 2026-27 format
    info = parse_agricultural_year("2026-27")
    assert info["start_year"] == 2026
    assert info["end_year"] == 2027
    assert info["formatted"] == "2026-27"
    assert info["year_num"] == 2026.0
    assert info["is_future"] is True

    # Test 2030-31 format
    info2 = parse_agricultural_year("2030-31")
    assert info2["start_year"] == 2030
    assert info2["end_year"] == 2031
    assert info2["formatted"] == "2030-31"

    # Test single 4-digit year e.g. 2028
    info3 = parse_agricultural_year("2028")
    assert info3["start_year"] == 2028
    assert info3["end_year"] == 2029
    assert info3["formatted"] == "2028-29"


def test_predict_farm_production_matches_yield_times_area(client):
    """Regression test: hero predicted production must equal predicted_yield * area and match financial production."""
    import re

    area = 5.0
    expected_yield = predictor.predict_yield(
        state="Punjab",
        district="Ludhiana",
        crop="Wheat",
        year="2026-27",
        season="Rabi",
        area=area,
    )
    expected_farm_prod = round(expected_yield * area, 2)
    expected_quintals = round(expected_farm_prod * 10.0, 2)

    # Confirm the district-aggregate production_model output is much larger and NOT shown as farm production
    district_agg_prod = round(
        predictor.predict(
            state="Punjab",
            district="Ludhiana",
            crop="Wheat",
            year="2026-27",
            season="Rabi",
            area=area,
        ),
        2,
    )
    assert district_agg_prod > expected_farm_prod * 10

    response = client.post("/predict", data={
        "state": "Punjab",
        "district": "Ludhiana",
        "crop": "Wheat",
        "year": "2026-27",
        "season": "Rabi",
        "area": str(area),
    })
    assert response.status_code == 200
    content = response.data.decode("utf-8")

    # Extract hero <h2> inside result-card
    m = re.search(r'<div class="result-card">.*?<h2>\s*([0-9,\.]+)\s*</h2>', content, re.DOTALL)
    assert m is not None, "Hero production <h2> not found in /predict response"
    displayed_prod = float(m.group(1).replace(",", ""))

    assert displayed_prod == expected_farm_prod
    assert displayed_prod != district_agg_prod
    assert f"{expected_quintals:,.1f} Quintals" in content


def test_predict_zero_and_invalid_area_inputs(client):
    """Regression test: zero, negative, and non-numeric area inputs are rejected safely."""
    for bad_area in ["0", "0.0", "-5", "abc", "nan", "inf"]:
        response = client.post("/predict", data={
            "state": "Punjab",
            "district": "Ludhiana",
            "crop": "Wheat",
            "year": "2026-27",
            "season": "Rabi",
            "area": bad_area,
        })
        assert response.status_code == 200
        content = response.data.decode("utf-8")
        assert "Prediction Error" in content
        assert "PREDICTED PRODUCTION" not in content


def test_recommend_renders_future_year_disclaimer(client):
    """Task 3 regression test: /recommend must render the honest future-year planning disclaimer."""
    response = client.post("/recommend", data={
        "state": "Punjab",
        "district": "Ludhiana",
        "season": "Rabi",
        "year": "2026-27",
        "area": "5.0",
    })
    assert response.status_code == 200
    content = response.data.decode("utf-8")
    assert "Model Note:" in content
    assert "Predictions do not vary across future agricultural years" in content


def test_weather_unobserved_district_does_not_fabricate_zero(client):
    """Task 4 regression test: /weather displays Unobserved/N/A instead of 0 mm or 0% when missing."""
    response = client.post("/weather", data={
        "state": "Punjab",
        "district": "NonExistentDistrict999",
        "season": "Kharif",
    })
    assert response.status_code == 200
    content = response.data.decode("utf-8")
    assert "Unobserved" in content
    assert "No IMD station match" in content
    assert ">0 mm<" not in content
    assert ">0%<" not in content


def test_train_models_paths_and_chronological_split_config():
    """Task 5 regression test: verify train_models.py resolves project root paths and strict cutoff year."""
    from pathlib import Path
    from scripts.modeling import train_models as tm

    assert tm.PROJECT_ROOT == Path(PROJECT_DIR)
    assert tm.PRODUCTION_CSV.exists()
    assert tm.MODELS_DIR.exists()
    assert tm.TRAIN_CUTOFF_YEAR == 2015


def test_weather_post_with_explicit_crop(client):
    """Task 7 regression test: /weather evaluates water risk for the user-selected crop."""
    response = client.post("/weather", data={
        "state": "Andhra Pradesh",
        "district": "Kurnool",
        "season": "Kharif",
        "crop": "Gram",
    })
    assert response.status_code == 200
    content = response.data.decode("utf-8")
    assert "Kurnool" in content
    assert "Drought-hardy" in content or "Moisture Risk:" in content


def test_canonical_model_plot_route_serves_png(client):
    """Task 9 regression test: /models/plots/<filename> serves canonical diagnostic PNGs."""
    resp = client.get("/models/plots/yield_prediction_model_comparison.png")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
    assert len(resp.data) > 1000


def test_weather_page_renders_temperature_profile(client):
    """Task 10 regression test: /weather renders state-level IMD 1x1 deg temperature & heat-stress card."""
    resp = client.post("/weather", data={
        "state": "Punjab",
        "district": "Ludhiana",
        "season": "Rabi",
        "crop": "Wheat",
    })
    assert resp.status_code == 200
    content = resp.data.decode("utf-8")
    assert "Heat Stress:" in content
    assert "IMD 1°×1° Temperature Grid" in content
    assert "°C" in content


def test_predict_renders_per_market_financial_scenarios(client):
    """Task 11 regression test: /predict renders per-market Est. Revenue, Est. Profit, and ROI columns."""
    resp = client.post("/predict", data={
        "state": "Andhra Pradesh",
        "district": "Kurnool",
        "crop": "Gram",
        "year": "2026-27",
        "season": "Rabi",
        "area": "3.0",
    })
    assert resp.status_code == 200
    content = resp.data.decode("utf-8")
    assert "Per-Market Financial Scenarios" in content
    assert "Est. Revenue" in content
    assert "Est. Profit" in content
    assert "ROI %" in content


def test_dashboard_metrics_and_no_unintegrated_temperature_claim(client):
    """Task 12 regression test: /dashboard binds real metrics and does not claim unintegrated temperature rows."""
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    content = resp.data.decode("utf-8")
    assert "326,999" in content
    assert "3,880" in content
    assert "CACP Cost Surveys" in content
    assert "2,808,042" not in content


def test_yield_model_excludes_bales_and_nuts_crops():
    """Regression test: document that Bales/Nuts-only crops are out-of-distribution for Tonnes yield_model."""
    cat_enc = predictor.yield_model.named_steps["preprocessor"].named_transformers_["cat"]
    learned_crops = set(cat_enc.categories_[2])
    # Cotton(lint), Jute, Mesta (Bales) and Coconut (Nuts) have 0 Tonnes rows in training CSV
    for non_tonnes_crop in ["Cotton(lint)", "Jute", "Mesta", "Coconut"]:
        assert non_tonnes_crop not in learned_crops
    # Standard Tonnes crops must be present
    for tonnes_crop in ["Wheat", "Rice", "Gram", "Sugarcane"]:
        assert tonnes_crop in learned_crops


def test_negative_raw_yield_and_ood_input_handling(client):
    """Phase 2.1 & 2.3 regression test: unsupported Bales/Nuts crops are rejected clearly and promoted Tweedie model produces non-negative raw yields."""
    import math
    import pandas as pd
    import pytest
    from src.models.predictor import UnsupportedCropError

    # 1. All 4 non-Tonnes crops must raise UnsupportedCropError (never silently encode as -1)
    for bad_crop in ["Cotton(lint)", "Jute", "Mesta", "Coconut"]:
        with pytest.raises(UnsupportedCropError) as exc_info:
            predictor.predict_yield(
                state="Maharashtra",
                district="Aurangabad",
                crop=bad_crop,
                year="2026-27",
                season="Kharif",
                area=5.0,
            )
        assert "not supported" in str(exc_info.value).lower()

        # Verify POST /predict also rejects clearly without rendering a production card
        resp = client.post("/predict", data={
            "state": "Maharashtra",
            "district": "Aurangabad",
            "crop": bad_crop,
            "year": "2026-27",
            "season": "Kharif",
            "area": "5.0",
        })
        assert resp.status_code == 200
        html = resp.data.decode("utf-8")
        assert "Prediction Error" in html
        assert "not supported" in html.lower()
        assert "PREDICTED PRODUCTION" not in html

    # 2. Previously negative in-distribution case (Kerala / WAYANAD / Sesamum / Whole Year)
    # must now produce a strictly positive raw prediction under promoted Tweedie model
    low_yield_row = pd.DataFrame([{
        "State": "Kerala",
        "District": "WAYANAD",
        "Crop": "Sesamum",
        "Season": "Whole Year",
        "Year_Num": 2018.0,
    }])
    raw_pred = float(predictor.yield_model.predict(low_yield_row)[0])
    assert math.isfinite(raw_pred)
    assert raw_pred > 0.0


def test_yield_predictions_across_small_medium_large_areas():
    """Phase 2.2 regression test: yield per hectare must be strictly invariant to farmer plot area."""
    import math

    for state, district, crop, season in [
        ("Punjab", "Ludhiana", "Wheat", "Rabi"),
        ("Andhra Pradesh", "Kurnool", "Gram", "Rabi"),
        ("Andhra Pradesh", "Kurnool", "Rice", "Kharif"),
        ("Uttar Pradesh", "Lucknow", "Sugarcane", "Whole Year"),
    ]:
        yields = []
        for area in [1.0, 5.0, 100.0, 1000.0, 50000.0]:
            y = predictor.predict_yield(
                state=state,
                district=district,
                crop=crop,
                year="2026-27",
                season=season,
                area=area,
            )
            assert math.isfinite(y)
            assert y > 0.0
            yields.append(y)
        # All areas must produce the exact same yield per hectare
        assert max(yields) - min(yields) < 1e-6, f"Yield varied by area for {state}/{district}/{crop}: {yields}"
