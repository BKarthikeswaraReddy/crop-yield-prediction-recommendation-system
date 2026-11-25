# Machine Learning Based Crop Yield Prediction & Recommendation System

An end-to-end agricultural analytics and farmer decision-support web application built with **Flask**, **scikit-learn**, **XGBoost**, and **pandas/PyArrow**, integrating official Indian government datasets for crop yield prediction, multi-criteria crop recommendation, APMC mandi market intelligence, CACP cultivation costs, and IMD weather risk assessment.

---

## 1. Problem Statement & Objectives

Farmers and agricultural planners frequently face uncertainty around expected crop yields, local wholesale prices, cultivation costs, and seasonal weather risks. This project addresses two primary objectives:

1. **Machine Learning Based Crop Yield Prediction:** Predict crop yield (`tonnes/hectare`) independently of farmer plot size and derive physically consistent farm-level production (`tonnes` and `quintals`) for a chosen State, District, Crop, Season, and Cultivated Area (hectares).
2. **Machine Learning Based Crop Recommendation:** Evaluate and rank historically cultivated crops for a district and season using a balanced multi-criteria suitability index combining economic returns (35%), weather resilience (30%), data certainty (20%), and yield reliability (15%).

Supporting intelligence modules include **APMC Mandi Price Intelligence** (with per-market financial scenarios), **CACP $C_2$ Cultivation Cost Benchmarking**, **Farm Financial Economics**, and **IMD Rainfall & Regional Temperature Intelligence**.

---

## 2. System Architecture & Technology Stack

- **Backend:** Python 3.11, Flask 3.1, Werkzeug, Gunicorn
- **Machine Learning:** XGBoost 3.2 (`reg:tweedie` p=1.5 for Yield; `reg:squarederror` for district Production), scikit-learn 1.9.0 (`ColumnTransformer`, `OrdinalEncoder`, `StandardScaler`, `Pipeline`), Joblib
- **Data Engineering:** pandas 3.1, NumPy 2.4, PyArrow 26.0 (Parquet), OpenPyXL
- **Frontend:** Jinja2 HTML5 templates, responsive custom CSS (`src/static/style.css`), vanilla JavaScript cascading dropdowns, static Matplotlib diagnostic plots served from canonical `models/plots/`
- **Testing & CI:** pytest 9.2, pytest-cov 7.1, GitHub Actions (`.github/workflows/ci.yml`)

---

## 3. Verified Datasets & Preprocessing

| Dataset | Runtime File | Scope & Units | Preprocessing & Notes |
|---|---|---|---|
| **India Agriculture Crop Production** | `data/India Agriculture Crop Production.csv` | 345,407 raw rows (`326,999` usable Tonnes rows), 36 States/UTs, 728 districts, `1997-98`–`2020-21`, Area (ha), Production (Tonnes) | Filtered strictly to `Tonnes`, `Area > 0`, `Production >= 0`; 99.9th percentile outlier cap fit on training data only (`<= 2015`). Non-Tonnes crops (`Cotton(lint)`, `Jute`, `Mesta` in Bales; `Coconut` in Nuts) are explicitly rejected rather than silently encoded as unknown `-1`. |
| **CACP Cost of Cultivation** | `data/Cost_of_Cultivation/cost-of-cultivation-cleaned.csv` | 3,880 official state-crop surveys (`1996-97`–`2021-22`), $C_2$ cost (`₹/ha` and `₹/quintal`) | Chronologically sorted by survey year so fallback queries always retrieve the latest official survey (`2021-22`); state-level with transparent national benchmark fallback. |
| **Agmarknet Mandi Prices** | `data/mandi_crop_{district,market,state,national}_summary.parquet`, `data/mandi_market_detailed_summary.parquet` | Aggregated from 29,699,148 APMC transactions (`2001`–`2026`), `₹/quintal` | Mapped via `data/crop_name_mapping.csv`; tiered District → State → National → Unavailable fallback plus individual APMC market records (commodity, variety, grade, arrival date). |
| **IMD District Rainfall** | `data/weather/rainfall_district_season_summary.parquet` | 5,379 canonical rows across 37 normalized States/UTs (`2018`–`2026`), mm & CV% | Canonical state/district normalization eliminates 2026 casing duplicates; state-disambiguated lookups prevent cross-state collisions (e.g., Aurangabad in Bihar vs. Maharashtra). |
| **IMD Gridded Temperature** | `data/weather/temperature_monthly_summary.parquet` | 34,062 rows on 1°×1° IMD grid (`2018`–`2025`), °C | Aggregated at **State-level** bounding boxes with heat-stress indicators; unmapped UTs return `Unobserved` (never fabricated). |

---

## 4. Machine Learning Models & Holdout Evaluation

Models were evaluated on a strict chronological holdout split (**Train:** `1997–2015`, `257,050` rows; **Holdout Test:** `2016–2020`, `69,636` rows) with zero target leakage:

| Target | Selected Algorithm | Features | Holdout $R^2$ | Holdout MAE | Holdout RMSE | Raw Negative Holdout Predictions |
|---|---|---|---:|---:|---:|---:|
| **Crop Yield (t/ha)** | **XGBoost (`reg:tweedie`, `p=1.5`)** | `State`, `District`, `Crop`, `Season`, `Year_Num` *(Area excluded)* | **`0.8998`** | **`1.2885 t/ha`** | **`3.9658 t/ha`** | **`0` (`0.00%`)** |
| **District Production (t)** | XGBoost (`reg:squarederror`) | `State`, `District`, `Crop`, `Season`, `Area`, `Year_Num` | `0.9618` | `10,450.22 t` | `79,195.04 t` | District-scale reference only |

### Key Yield Model Properties
- **Mathematically Non-Negative:** Tweedie log-link (`p=1.5`) handles zero-yield training observations natively and produces strictly positive predictions (`min = +0.1131 t/ha` across all 69,636 holdout records, compared to `1,457` negative predictions in the previous `reg:squarederror` baseline).
- **Area-Independent Farm Yield:** Excluding district-aggregate `Area` ensures predicted yield per hectare does not change merely because a farmer enters `1 ha`, `5 ha`, or `1,000 ha`.
- **Unsupported Crop Guard:** `Cotton(lint)`, `Jute`, `Mesta`, and `Coconut` raise an explicit user-facing message explaining the `Bales`/`Nuts` unit mismatch instead of silently encoding them as `-1`.

---

## 5. Financial Calculation Methodology

All farm-level economics use verified, unit-consistent formulas:

- $\text{Farm Production (Tonnes)} = \text{Predicted Yield (t/ha)} \times \text{Cultivated Area (ha)}$
- $\text{Farm Production (Quintals)} = \text{Farm Production (Tonnes)} \times 10$ *(1 metric tonne = 10 quintals)*
- $\text{Gross Revenue (₹)} = \text{Farm Production (Quintals)} \times \text{Mandi Modal Price (₹/quintal)}$
- $\text{Total Cultivation Cost (₹)} = \text{CACP } C_2 \text{ Cost (₹/ha)} \times \text{Cultivated Area (ha)}$
- $\text{Estimated Profit / Loss (₹)} = \text{Gross Revenue} - \text{Total Cultivation Cost}$
- $\text{ROI (\%)} = \left(\frac{\text{Estimated Profit}}{\text{Total Cultivation Cost}}\right) \times 100$ *(computed only when Cost > 0)*

Per-market scenarios on `/predict` compute Revenue, Profit, and ROI independently for each local APMC market record. Missing prices or costs always evaluate to `None` / `N/A` and are never fabricated as zero.

---

## 6. Installation & Execution

```bash
# 1. Clone the repository
git clone https://github.com/BKarthikeswaraReddy/crop-yield-prediction-recommendation-system.git
cd crop-yield-prediction-recommendation-system

# 2. Create and activate a Python 3.11 virtual environment
python -m venv .venv_clean
# Windows (PowerShell):
.venv_clean\Scripts\Activate.ps1
# Linux / macOS:
source .venv_clean/bin/activate

# 3. Install pinned dependencies (includes scikit-learn==1.9.0)
pip install -r requirements.txt

# 4. Run the Flask application
python run.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 7. Web Routes & API Endpoints

| Route | Method | Purpose |
|---|---|---|
| `/` | `GET` | Crop Yield Prediction & Financial Analysis form |
| `/predict` | `POST` | Farm-level yield/production, financial analysis, per-market APMC scenarios, weather risk |
| `/recommend` | `GET`, `POST` | Multi-criteria Crop Recommendation Advisor |
| `/weather` | `GET`, `POST` | State-disambiguated IMD seasonal rainfall & state-level 1°×1° temperature/heat-stress profile |
| `/dashboard` | `GET` | Platform overview, live record counts, model performance, and national mandi benchmarks |
| `/analytics` | `GET` | 5-algorithm comparison table and diagnostic plots |
| `/models/plots/<filename>` | `GET` | Serves canonical diagnostic PNGs from `models/plots/` |
| `/api/districts?state=...` | `GET` | Authoritative state → district JSON mapping (post-bifurcation AP & Telangana) |
| `/api/price?state=...&district=...&crop=...` | `GET` | Tiered APMC modal price JSON |
| `/api/mandi_markets?state=...&district=...&crop=...` | `GET` | Individual APMC market records JSON |

---

## 8. Testing & Coverage

Run the full automated test suite (72 tests, 0 warnings):

```bash
.venv_clean\Scripts\python -m pytest tests -v
.venv_clean\Scripts\python -m pytest tests --cov=src --cov-report=term-missing
```

GitHub Actions CI (`.github/workflows/ci.yml`) automatically runs the test suite and coverage check on Python `3.11` and `3.12` using only Git-tracked runtime datasets.

---

## 9. Limitations & Future Scope

- **Planning Benchmarks, Not Autoregressive Forecasts:** Tree-based models cannot extrapolate trends beyond their `1997–2020` training ceiling; future agricultural years (`2026-27`–`2035-36`) return the regional historical planning benchmark (explicitly disclosed on both `/` and `/recommend`).
- **Non-Tonnes Crops:** `Cotton(lint)`, `Jute`, `Mesta` (`Bales`), and `Coconut` (`Nuts`) require dedicated unit-consistent models in a future release.
- **Spatial Resolution of Weather & Costs:** CACP $C_2$ costs (`2021-22`) and IMD 1°×1° temperature profiles (`2018–2025`) are state-level benchmarks; wholesale APMC modal prices are historical reference benchmarks, not guaranteed farmgate prices.
- **Future Scope:** Integrate soil NPK/pH and seasonal rainfall/temperature directly into yield training features, add district boundary shapefiles for district-level temperature, and add unit-specific models for Bales/Nuts crops.

---

## License

MIT License
