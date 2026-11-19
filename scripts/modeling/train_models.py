"""
ML Model Training Pipeline
Crop Production Prediction and Farmer Decision Support System

Trains and compares regression models for:
  A. Crop Production Prediction (target: Production in tonnes)
  B. Crop Yield Prediction (target: Yield in tonnes/hectare)

Algorithms compared:
  1. Linear Regression
  2. Decision Tree Regressor
  3. Random Forest Regressor
  4. Gradient Boosting Regressor
  5. XGBoost Regressor (if available)

Saves best models, preprocessing pipelines, metrics, and visualizations.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

# Try XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# -------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
MODELS_DIR = SCRIPT_DIR / "models"
PLOTS_DIR = SCRIPT_DIR / "models" / "plots"

MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Input data
PRODUCTION_CSV = DATA_DIR / "India Agriculture Crop Production.csv"
PRODUCTION_COST_CSV = DATA_DIR / "production_cost_integrated.csv"


def load_and_prepare_data():
    """Load crop production data and prepare features/targets."""
    print("Loading dataset...")

    # Use original dataset for broader coverage
    df = pd.read_csv(PRODUCTION_CSV)
    print(f"Raw records: {len(df):,}")

    # Filter to Tonnes only
    df = df[df["Production Units"] == "Tonnes"].copy()

    # Convert numerics
    df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
    df["Production"] = pd.to_numeric(df["Production"], errors="coerce")

    # Extract numeric year from agricultural year (e.g., "2015-16" -> 2015)
    df["Year_Num"] = df["Year"].astype(str).str.extract(r"(\d{4})").astype(float)

    # Drop rows with missing essential values
    df = df.dropna(subset=["State", "District", "Crop", "Year", "Season", "Area", "Production", "Year_Num"])

    # Remove zero/negative area and production
    df = df[(df["Area"] > 0) & (df["Production"] >= 0)].copy()

    # Remove infinities
    df = df[np.isfinite(df["Area"]) & np.isfinite(df["Production"])].copy()

    # Compute yield
    df["Yield"] = df["Production"] / df["Area"]

    # Remove extreme yield outliers (top 0.1%)
    yield_99_9 = df["Yield"].quantile(0.999)
    df = df[df["Yield"] <= yield_99_9].copy()

    # Clean string columns
    for col in ["State", "District", "Crop", "Season"]:
        df[col] = df[col].astype(str).str.strip()

    print(f"Usable records after filtering: {len(df):,}")
    print(f"States: {df['State'].nunique()}, Districts: {df['District'].nunique()}, Crops: {df['Crop'].nunique()}")
    print(f"Year range: {df['Year_Num'].min():.0f} - {df['Year_Num'].max():.0f}")
    print(f"Production range: {df['Production'].min():.0f} - {df['Production'].max():.0f} tonnes")
    print(f"Yield range: {df['Yield'].min():.2f} - {df['Yield'].max():.2f} t/ha")

    return df


def get_models():
    """Return dict of model name -> model instance."""
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=20, min_samples_leaf=5, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=20, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=80, max_depth=5, learning_rate=0.1,
            min_samples_leaf=5, random_state=42
        ),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=150, max_depth=8, learning_rate=0.1,
            min_child_weight=5, random_state=42, n_jobs=-1,
            verbosity=0
        )
    return models


def build_preprocessor(categorical_features, numerical_features):
    """Build sklearn ColumnTransformer."""
    return ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             categorical_features),
            ("num", StandardScaler(), numerical_features),
        ],
        remainder="drop"
    )


def train_and_compare(df, target_col, target_name, categorical_features,
                      numerical_features, models_dict):
    """Train all models for one target, compare, and return results."""
    print(f"\n{'=' * 85}")
    print(f"TRAINING: {target_name}")
    print(f"Target: {target_col}")
    print(f"{'=' * 85}")

    X = df[categorical_features + numerical_features].copy()
    y = df[target_col].copy()

    # Chronological split using Year_Num
    if "Year_Num" in df.columns:
        # Sort by year for chronological split
        sorted_idx = np.argsort(df["Year_Num"].values)
        X_sorted = X.iloc[sorted_idx]
        y_sorted = y.iloc[sorted_idx]
        split_point = int(len(X_sorted) * 0.8)
        X_train = X_sorted.iloc[:split_point]
        X_test = X_sorted.iloc[split_point:]
        y_train = y_sorted.iloc[:split_point]
        y_test = y_sorted.iloc[split_point:]
        print(f"Chronological split: Train={len(X_train):,} (earlier years), Test={len(X_test):,} (later years)")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Random split: Train={len(X_train):,}, Test={len(X_test):,}")

    results = []
    best_model = None
    best_r2 = -float("inf")
    best_name = ""
    best_pipeline = None

    for name, model in models_dict.items():
        print(f"\n  Training {name}...")
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(categorical_features, numerical_features)),
            ("regressor", model),
        ])

        pipeline.fit(X_train, y_train)
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        # Metrics on TEST set
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2 = r2_score(y_test, y_pred_test)

        # Training metrics for comparison
        r2_train = r2_score(y_train, y_pred_train)

        result = {
            "model": name,
            "target": target_name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2_test": round(r2, 4),
            "R2_train": round(r2_train, 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        results.append(result)

        print(f"    MAE={mae:,.2f} | RMSE={rmse:,.2f} | R²(test)={r2:.4f} | R²(train)={r2_train:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_model = model
            best_pipeline = pipeline
            best_y_test = y_test.values
            best_y_pred = y_pred_test

    print(f"\n  ★ Best model: {best_name} (R²={best_r2:.4f})")

    # Generate plots for best model
    safe_target = target_name.lower().replace(" ", "_")
    plot_actual_vs_predicted(best_y_test, best_y_pred, best_name, target_name, safe_target)
    plot_residuals(best_y_test, best_y_pred, best_name, target_name, safe_target)

    # Feature importance for tree-based models
    if hasattr(best_pipeline.named_steps["regressor"], "feature_importances_"):
        plot_feature_importance(
            best_pipeline, categorical_features, numerical_features,
            best_name, target_name, safe_target
        )

    return results, best_pipeline, best_name


def plot_actual_vs_predicted(y_true, y_pred, model_name, target_name, safe_name):
    """Actual vs Predicted scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sample_size = min(5000, len(y_true))
    idx = np.random.RandomState(42).choice(len(y_true), sample_size, replace=False)

    ax.scatter(y_true[idx], y_pred[idx], alpha=0.3, s=10, c="#2196F3")
    max_val = max(y_true[idx].max(), y_pred[idx].max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{target_name} — Actual vs Predicted\n({model_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{safe_name}_actual_vs_predicted.png", dpi=100)
    plt.close(fig)
    print(f"    Plot saved: {safe_name}_actual_vs_predicted.png")


def plot_residuals(y_true, y_pred, model_name, target_name, safe_name):
    """Residual distribution plot."""
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sample_size = min(5000, len(residuals))
    idx = np.random.RandomState(42).choice(len(residuals), sample_size, replace=False)

    ax1.scatter(y_pred[idx], residuals[idx], alpha=0.3, s=10, c="#FF9800")
    ax1.axhline(y=0, color="r", linestyle="--")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Residual")
    ax1.set_title(f"Residuals vs Predicted")

    ax2.hist(residuals, bins=100, color="#4CAF50", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residual Distribution")

    fig.suptitle(f"{target_name} Residuals ({model_name})", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{safe_name}_residuals.png", dpi=100)
    plt.close(fig)
    print(f"    Plot saved: {safe_name}_residuals.png")


def plot_feature_importance(pipeline, cat_features, num_features, model_name, target_name, safe_name):
    """Feature importance bar chart."""
    regressor = pipeline.named_steps["regressor"]
    preprocessor = pipeline.named_steps["preprocessor"]

    importances = regressor.feature_importances_

    # Get feature names
    try:
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_names = list(cat_features)
    except Exception:
        cat_names = list(cat_features)

    feature_names = cat_names + list(num_features)

    # For OrdinalEncoder, importances align with feature count directly
    if len(importances) == len(feature_names):
        feat_imp = dict(zip(feature_names, importances))
    else:
        # Aggregate or just show top
        feat_imp = {f"Feature_{i}": imp for i, imp in enumerate(importances)}
        # Just show top 20
        sorted_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:20]
        feat_imp = dict(sorted_imp)

    sorted_items = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items[:15]]
    values = [x[1] for x in sorted_items[:15]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(names)), values, color="#2196F3", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"{target_name} — Feature Importance ({model_name})")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{safe_name}_feature_importance.png", dpi=100)
    plt.close(fig)
    print(f"    Plot saved: {safe_name}_feature_importance.png")


def plot_model_comparison(all_results):
    """Bar chart comparing all models on R², RMSE, MAE."""
    df = pd.DataFrame(all_results)

    for target_name in df["target"].unique():
        sub = df[df["target"] == target_name].sort_values("R2_test", ascending=False)
        safe_name = target_name.lower().replace(" ", "_")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # R²
        axes[0].barh(sub["model"], sub["R2_test"], color="#4CAF50", alpha=0.8)
        axes[0].set_xlabel("R² Score (Test)")
        axes[0].set_title("R² Comparison")
        axes[0].invert_yaxis()

        # RMSE
        axes[1].barh(sub["model"], sub["RMSE"], color="#FF9800", alpha=0.8)
        axes[1].set_xlabel("RMSE")
        axes[1].set_title("RMSE Comparison")
        axes[1].invert_yaxis()

        # MAE
        axes[2].barh(sub["model"], sub["MAE"], color="#2196F3", alpha=0.8)
        axes[2].set_xlabel("MAE")
        axes[2].set_title("MAE Comparison")
        axes[2].invert_yaxis()

        fig.suptitle(f"Model Comparison — {target_name}", fontsize=13)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{safe_name}_model_comparison.png", dpi=100)
        plt.close(fig)
        print(f"  Plot saved: {safe_name}_model_comparison.png")


def main():
    print("=" * 85)
    print("ML MODEL TRAINING PIPELINE")
    print("Crop Production Prediction and Farmer Decision Support System")
    print("=" * 85)

    df = load_and_prepare_data()

    categorical_features = ["State", "District", "Crop", "Season"]
    numerical_features = ["Area", "Year_Num"]

    models_dict = get_models()
    print(f"\nModels to compare: {list(models_dict.keys())}")
    print(f"XGBoost available: {HAS_XGBOOST}")

    all_results = []

    # A. Production prediction
    prod_results, prod_pipeline, prod_best = train_and_compare(
        df, "Production", "Production Prediction",
        categorical_features, numerical_features, get_models()
    )
    all_results.extend(prod_results)

    # B. Yield prediction
    yield_results, yield_pipeline, yield_best = train_and_compare(
        df, "Yield", "Yield Prediction",
        categorical_features, numerical_features, get_models()
    )
    all_results.extend(yield_results)

    # Model comparison plots
    print(f"\n{'=' * 85}")
    print("GENERATING MODEL COMPARISON PLOTS")
    print(f"{'=' * 85}")
    plot_model_comparison(all_results)

    # Save best models
    print(f"\n{'=' * 85}")
    print("SAVING MODELS AND METADATA")
    print(f"{'=' * 85}")

    # Save production model
    prod_model_path = MODELS_DIR / "production_model.joblib"
    joblib.dump(prod_pipeline, prod_model_path)
    print(f"Production model saved: {prod_model_path}")
    print(f"  Best algorithm: {prod_best}")

    # Save yield model
    yield_model_path = MODELS_DIR / "yield_model.joblib"
    joblib.dump(yield_pipeline, yield_model_path)
    print(f"Yield model saved: {yield_model_path}")
    print(f"  Best algorithm: {yield_best}")

    # Save metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "dataset": str(PRODUCTION_CSV),
        "total_records": len(df),
        "features": {
            "categorical": categorical_features,
            "numerical": numerical_features,
        },
        "production_model": {
            "best_algorithm": prod_best,
            "model_path": str(prod_model_path),
        },
        "yield_model": {
            "best_algorithm": yield_best,
            "model_path": str(yield_model_path),
        },
        "all_results": all_results,
        "dataset_stats": {
            "states": df["State"].nunique(),
            "districts": df["District"].nunique(),
            "crops": df["Crop"].nunique(),
            "seasons": sorted(df["Season"].unique().tolist()),
            "year_range": [int(df["Year_Num"].min()), int(df["Year_Num"].max())],
            "unique_state_list": sorted(df["State"].unique().tolist()),
            "unique_crop_list": sorted(df["Crop"].unique().tolist()),
        },
        "xgboost_available": HAS_XGBOOST,
    }

    # Save feature names for Streamlit
    metadata["feature_columns"] = categorical_features + numerical_features

    metrics_path = MODELS_DIR / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metrics saved: {metrics_path}")

    # Save dataset options for Streamlit (unique values for dropdowns)
    options = {
        "states": sorted(df["State"].unique().tolist()),
        "districts": {
            state: sorted(df[df["State"] == state]["District"].unique().tolist())
            for state in df["State"].unique()
        },
        "crops": sorted(df["Crop"].unique().tolist()),
        "seasons": sorted(df["Season"].unique().tolist()),
        "years": sorted(df["Year_Num"].unique().astype(int).tolist()),
        "crop_seasons": {
            crop: sorted(df[df["Crop"] == crop]["Season"].unique().tolist())
            for crop in df["Crop"].unique()
        },
    }
    options_path = MODELS_DIR / "dataset_options.json"
    with open(options_path, "w") as f:
        json.dump(options, f, indent=2)
    print(f"Dataset options saved: {options_path}")

    # Print final summary
    print(f"\n{'=' * 85}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 85}")

    results_df = pd.DataFrame(all_results)
    print("\nAll Model Results:")
    print(results_df.to_string(index=False))

    # Save results CSV
    results_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    print(f"\nResults CSV: {MODELS_DIR / 'model_comparison.csv'}")

    return metadata


if __name__ == "__main__":
    main()
