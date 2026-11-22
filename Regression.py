import logging
import pandas as pd
import joblib

from Load import load_data
from Transform import numerical_columns, categorical_columns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from config import (
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


def load_training_data():
    """Load transformed training data and separate features from target."""

    train_path = PROCESSED_DATA_DIR / "train_transformed.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    df = pd.read_csv(train_path)

    if df.empty:
        raise ValueError("Training dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Training data contains missing values.")

    logger.info("Training data loaded successfully.")
    logger.info("Features shape: %s", X.shape)
    logger.info("Target shape: %s", y.shape)

    return X, y


def train_linear_regression(X, y):
    """Train a Linear Regression model."""

    logger.info("Training Linear Regression model...")

    model = LinearRegression()
    model.fit(X, y)

    logger.info("Linear Regression training completed successfully.")

    return model


def train_random_forest(X, y):
    """Train a Random Forest Regressor with reproducible settings."""

    logger.info("Training Random Forest Regressor model...")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X, y)

    logger.info("Random Forest Regressor training completed successfully.")

    return model


def save_model(model, filename):
    """Save a trained regression model for later use."""

    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / filename

    joblib.dump(model, model_path)

    logger.info("Model saved successfully: %s", model_path)

    return model_path



def compare_models():
    """Compare models using leakage-safe preprocessing and validation."""

    logger.info("Starting preliminary model comparison...")

    # Load raw data without fitting preprocessing on validation data
    df = load_data()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    # Remove invalid target values
    df = df.copy()
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="coerce"
    )
    df = df[df[TARGET_COLUMN].between(0, 100)]

    feature_columns = numerical_columns + categorical_columns
    X_raw = df[feature_columns]
    y_raw = df[TARGET_COLUMN]

    # Reproduce the original train/test split
    X_train_full, _, y_train_full, _ = train_test_split(
        X_raw,
        y_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Split the original training portion into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ])

    models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

    results = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_val)

        mae = mean_absolute_error(y_val, predictions)
        rmse = mean_squared_error(y_val, predictions) ** 0.5
        r2 = r2_score(y_val, predictions)

        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

        logger.info(
            "%s | MAE: %.4f | RMSE: %.4f | R2: %.4f",
            name, mae, rmse, r2
        )

    logger.info("Preliminary model comparison completed.")
    return results


if __name__ == "__main__":
    X, y = load_training_data()

    comparison_results = compare_models()

    linear_model = train_linear_regression(X, y)
    forest_model = train_random_forest(X, y)

    save_model(linear_model, "linear_regression.joblib")
    save_model(forest_model, "random_forest_regressor.joblib")