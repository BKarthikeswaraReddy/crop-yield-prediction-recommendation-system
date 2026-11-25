import os
import re
import json
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.year_utils import parse_agricultural_year


class UnsupportedCropError(ValueError):
    """Raised when a requested crop is not supported by the Tonnes-based yield model."""


# Crops whose official dataset records are exclusively in non-Tonnes units
UNSUPPORTED_NON_TONNES_CROPS = {
    "COTTON(LINT)": ("Cotton(lint)", "Bales"),
    "COTTON": ("Cotton(lint)", "Bales"),
    "JUTE": ("Jute", "Bales"),
    "MESTA": ("Mesta", "Bales"),
    "COCONUT": ("Coconut", "Nuts"),
}


class Predictor:

    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.production_model = None
        self.yield_model = None
        self.data = None
        self.metrics = None

        # Base directories for serialized models
        self.project_dir = Path(__file__).resolve().parent.parent.parent
        self.models_dir = self.project_dir / "models"
        self.prod_model_path = self.models_dir / "production_model.joblib"
        self.yield_model_path = self.models_dir / "yield_model.joblib"
        self.metrics_path = self.models_dir / "model_metrics.json"

        self.load_data()
        self._load_serialized_models()

    def _load_serialized_models(self):
        """Load pre-trained production and yield models and metrics if available."""
        if self.prod_model_path.exists():
            try:
                self.production_model = joblib.load(self.prod_model_path)
                self.model = self.production_model
                print(f"Loaded serialized production model from {self.prod_model_path}")
            except Exception as e:
                print(f"Warning: Failed to load serialized production model: {e}")

        if self.yield_model_path.exists():
            try:
                self.yield_model = joblib.load(self.yield_model_path)
                print(f"Loaded serialized yield model from {self.yield_model_path}")
            except Exception as e:
                print(f"Warning: Failed to load serialized yield model: {e}")

        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, "r") as f:
                    data = json.load(f)
                    # Find production model test metrics from all_results
                    for res in data.get("all_results", []):
                        if res.get("target") == "Production Prediction" and res.get("model") == data.get("production_model", {}).get("best_algorithm"):
                            self.metrics = {
                                "MAE": res.get("MAE"),
                                "RMSE": res.get("RMSE"),
                                "R2 Score": res.get("R2_test"),
                                "Algorithm": res.get("model")
                            }
                            break
            except Exception as e:
                print(f"Warning: Failed to load metrics: {e}")

    def load_data(self):
        """Load and prepare the crop production dataset."""

        self.data = pd.read_csv(self.data_path)

        required_columns = [
            'State', 'District', 'Crop', 'Year',
            'Season', 'Area', 'Production', 'Production Units'
        ]

        missing_columns = [
            col for col in required_columns
            if col not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        # Use only records measured in Tonnes
        self.data = self.data[
            self.data['Production Units'] == 'Tonnes'
        ].copy()

        # Convert numerical columns
        self.data['Area'] = pd.to_numeric(
            self.data['Area'], errors='coerce'
        )

        self.data['Production'] = pd.to_numeric(
            self.data['Production'], errors='coerce'
        )

        # Remove records with missing essential values
        self.data = self.data.dropna(
            subset=[
                'State', 'District', 'Crop',
                'Year', 'Season', 'Area', 'Production'
            ]
        )

        # Remove invalid infinite values
        self.data = self.data[
            self.data['Area'].map(lambda x: pd.notna(x) and abs(x) != float('inf'))
            & self.data['Production'].map(
                lambda x: pd.notna(x) and abs(x) != float('inf')
            )
        ]

        # Keep valid non-negative area and production
        self.data = self.data[
            (self.data['Area'] >= 0)
            & (self.data['Production'] >= 0)
        ].copy()

        # Treat year and other categorical fields consistently
        categorical_columns = [
            'State', 'District', 'Crop', 'Year', 'Season'
        ]

        for col in categorical_columns:
            self.data[col] = self.data[col].astype(str).str.strip()

        print("Dataset loaded successfully.")
        print("Usable records:", len(self.data))

    def train_model(self):
        """Train and evaluate a fallback Random Forest model."""

        categorical_features = [
            'State', 'District', 'Crop', 'Year', 'Season'
        ]

        numerical_features = ['Area']

        X = self.data[categorical_features + numerical_features]
        y = self.data['Production']

        # Encode categorical data and preserve numerical area
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    'categorical',
                    OneHotEncoder(handle_unknown='ignore'),
                    categorical_features
                ),
                ('numerical', 'passthrough', numerical_features)
            ]
        )

        self.model = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                (
                    'regressor',
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=25,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1
                    )
                )
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        print("\nTraining Random Forest model...")
        self.model.fit(X_train, y_train)

        # Evaluate using unseen test data
        predictions = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        self.metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2 Score': r2
        }

        print("\nMODEL EVALUATION RESULTS")
        print("------------------------")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")

        return self.metrics

    def predict(self, state, district, crop, year, season, area):
        """Predict crop production in Tonnes."""

        if self.model is None:
            self.train_model()

        year_info = parse_agricultural_year(year)
        year_num = year_info["year_num"]

        input_data = pd.DataFrame([{
            'State': str(state).strip(),
            'District': str(district).strip(),
            'Crop': str(crop).strip(),
            'Year': year_info["formatted"],
            'Season': str(season).strip(),
            'Area': float(area),
            'Year_Num': year_num
        }])

        prediction = self.model.predict(input_data)
        return max(0.0, float(prediction[0]))

    def get_supported_yield_crops(self):
        """Return the set of crop names learned by the serialized yield model."""
        if self.yield_model is None:
            return None
        pipeline = getattr(self.yield_model, "regressor_", self.yield_model)
        if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
            pre = pipeline.named_steps["preprocessor"]
            if hasattr(pre, "named_transformers_") and "cat" in pre.named_transformers_:
                cat_enc = pre.named_transformers_["cat"]
                if hasattr(cat_enc, "categories_") and len(cat_enc.categories_) >= 3:
                    return set(cat_enc.categories_[2])
        return None

    def validate_supported_crop(self, crop):
        """Validate that crop is supported by the Tonnes-based yield model (never silently encode as -1)."""
        crop_clean = str(crop).strip()
        crop_upper = crop_clean.upper()

        if crop_upper in UNSUPPORTED_NON_TONNES_CROPS:
            canonical, unit = UNSUPPORTED_NON_TONNES_CROPS[crop_upper]
            raise UnsupportedCropError(
                f"Crop '{canonical}' is not supported by the current Tonnes-based yield model "
                f"because its official dataset production unit is '{unit}', not Tonnes. "
                "No invented unit conversion is applied."
            )

        learned_crops = self.get_supported_yield_crops()
        if learned_crops is not None and crop_clean not in learned_crops:
            raise UnsupportedCropError(
                f"Crop '{crop_clean}' is not present in the validated Tonnes training dataset "
                "and cannot be reliably predicted by the current yield model."
            )

    def predict_yield(self, state, district, crop, year, season, area):
        """Predict crop yield in Tonnes/Hectare with explicit unsupported-crop validation."""
        self.validate_supported_crop(crop)

        year_info = parse_agricultural_year(year)
        year_num = year_info["year_num"]

        input_data = pd.DataFrame([{
            'State': str(state).strip(),
            'District': str(district).strip(),
            'Crop': str(crop).strip(),
            'Year': year_info["formatted"],
            'Season': str(season).strip(),
            'Area': float(area),
            'Year_Num': year_num
        }])

        if self.yield_model is not None:
            prediction = self.yield_model.predict(input_data)
            val = float(prediction[0])
            if not pd.notna(val) or abs(val) == float('inf'):
                raise ValueError("Yield model returned a non-finite value.")
            return max(0.0, val)
        else:
            # Fallback to production / area
            prod = self.predict(state, district, crop, year, season, area)
            return prod / float(area) if float(area) > 0 else 0.0
