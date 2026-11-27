
"""Evaluate the Regression Model using unseen test data."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config

logger = logging.getLogger(__name__)


def evaluate_regression_model(
    test_df=None,
    train_df=None,
    model_path=None,
):
    """Evaluate the trained regression model and compare it with a baseline.

    Args:
        test_df: Optional test DataFrame. If omitted, data is loaded
                 from test.csv.
        train_df: Optional training DataFrame used for fitting the
                  baseline model. If omitted, data is loaded from train.csv.
        model_path: Optional path to the saved regression model.

    Returns:
        Dictionary containing model and baseline evaluation metrics.
    """

    # Load test data when no DataFrame is provided.
    if test_df is None:
        test_path = Path(config.PROCESSED_DATA_DIR) / "test.csv"

        if not test_path.exists():
            raise FileNotFoundError(
                f"Test data file not found: {test_path}"
            )

        try:
            test_df = pd.read_csv(test_path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.exception("Failed to load test data")
            raise ValueError("Could not read the test data") from exc

    # Load training data when no DataFrame is provided.
    if train_df is None:
        train_path = Path(config.PROCESSED_DATA_DIR) / "train.csv"

        if not train_path.exists():
            raise FileNotFoundError(
                f"Training data file not found: {train_path}"
            )

        try:
            train_df = pd.read_csv(train_path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.exception("Failed to load training data")
            raise ValueError("Could not read the training data") from exc

    # Validate DataFrame inputs.
    if not isinstance(test_df, pd.DataFrame):
        raise TypeError("test_df must be a pandas DataFrame")

    if not isinstance(train_df, pd.DataFrame):
        raise TypeError("train_df must be a pandas DataFrame")

    if test_df.empty:
        raise ValueError("Test data cannot be empty")

    if train_df.empty:
        raise ValueError("Training data cannot be empty")

    target = config.TARGET_COLUMN

    if target not in test_df.columns:
        raise KeyError(
            f"Target column '{target}' is missing from test data"
        )

    if target not in train_df.columns:
        raise KeyError(
            f"Target column '{target}' is missing from training data"
        )

    # Validate test target.
    y_test = test_df[target].copy()

    if y_test.isna().any():
        raise ValueError("Test target contains missing values")

    if not pd.api.types.is_numeric_dtype(y_test):
        raise TypeError("Test target must be numeric")

    if not np.isfinite(y_test.to_numpy(dtype=float)).all():
        raise ValueError("Test target contains infinite values")

    # Validate training target.
    y_train = train_df[target].copy()

    if y_train.isna().any():
        raise ValueError("Training target contains missing values")

    if not pd.api.types.is_numeric_dtype(y_train):
        raise TypeError("Training target must be numeric")

    if not np.isfinite(y_train.to_numpy(dtype=float)).all():
        raise ValueError("Training target contains infinite values")

    # Separate test features and target.
    X_test = test_df.drop(columns=[target]).copy()

    if X_test.empty:
        raise ValueError("No test input features are available")

    if X_test.isna().any().any():
        raise ValueError("Test features contain missing values")

    non_numeric_columns = [
        col
        for col in X_test.columns
        if not pd.api.types.is_numeric_dtype(X_test[col])
    ]

    if non_numeric_columns:
        raise TypeError(
            f"All test features must be numeric: {non_numeric_columns}"
        )

    if not np.isfinite(X_test.to_numpy(dtype=float)).all():
        raise ValueError("Test features contain infinite values")

    # Load the saved regression model.
    if model_path is None:
        model_path = (
            Path(config.MODELS_DIR) / "regression_model.joblib"
        )
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Regression model not found: {model_path}"
        )

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        logger.exception("Failed to load regression model")
        raise ValueError(
            "Could not load the regression model"
        ) from exc

    # Check that test features match the model's expected features.
    if hasattr(model, "n_features_in_"):
        if X_test.shape[1] != model.n_features_in_:
            raise ValueError(
                "Test feature count does not match the trained model"
            )

    if hasattr(model, "feature_names_in_"):
        if list(X_test.columns) != list(model.feature_names_in_):
            raise ValueError(
                "Test feature names or order do not match the model"
            )

    # Evaluate the trained regression model.
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Train the baseline using training data only.
    baseline = DummyRegressor(strategy="mean")

    baseline.fit(
        np.zeros((len(y_train), 1)),
        y_train,
    )

    # Evaluate the baseline on the same test data.
    baseline_predictions = baseline.predict(
        np.zeros((len(y_test), 1))
    )

    baseline_mae = mean_absolute_error(
        y_test, baseline_predictions
    )
    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_predictions)
    )
    baseline_r2 = r2_score(y_test, baseline_predictions)

    # Store evaluation results.
    results = {
        "model": {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
        },
        "baseline": {
            "MAE": float(baseline_mae),
            "RMSE": float(baseline_rmse),
            "R2": float(baseline_r2),
        },
    }

    logger.info(
        "Regression evaluation completed on %d test rows",
        len(X_test),
    )

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    evaluation_results = evaluate_regression_model()

    print("\nRegression Evaluation Results")
    print("=" * 40)

    print("\nGradient Boosting Regressor:")
    for metric, value in evaluation_results["model"].items():
        print(f"{metric}: {value:.4f}")

    print("\nBaseline (Training Mean Prediction):")
    for metric, value in evaluation_results["baseline"].items():
        print(f"{metric}: {value:.4f}")
