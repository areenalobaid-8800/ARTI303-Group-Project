
"""Regression Model: predict students' Exam_Score."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

import config

logger = logging.getLogger(__name__)


def train_regression_model(train_df=None, model_path=None):
    """Train and save a Gradient Boosting regression model.

    Args:
        train_df: Optional training DataFrame.
                  If omitted, data is loaded from train.csv.
        model_path: Optional path for saving the trained model.

    Returns:
        The trained GradientBoostingRegressor.
    """

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

    # Validate the input.
    if not isinstance(train_df, pd.DataFrame):
        raise TypeError("train_df must be a pandas DataFrame")

    if train_df.empty:
        raise ValueError("Training data cannot be empty")

    target = config.TARGET_COLUMN

    if target not in train_df.columns:
        raise KeyError(f"Target column '{target}' is missing")

    if train_df[target].isna().any():
        raise ValueError("Target column contains missing values")

    if not pd.api.types.is_numeric_dtype(train_df[target]):
        raise TypeError("Regression target must be numeric")

    # Separate features and target.
    X_train = train_df.drop(columns=[target]).copy()
    y_train = train_df[target].copy()

    if X_train.empty:
        raise ValueError("No input features are available")

    if X_train.isna().any().any():
        raise ValueError("Input features contain missing values")

    non_numeric_columns = [
        col
        for col in X_train.columns
        if not pd.api.types.is_numeric_dtype(X_train[col])
    ]

    if non_numeric_columns:
        raise TypeError(
            f"All transformed features must be numeric: "
            f"{non_numeric_columns}"
        )

    if not np.isfinite(X_train.to_numpy(dtype=float)).all():
        raise ValueError("Input features contain infinite values")

    if not np.isfinite(y_train.to_numpy(dtype=float)).all():
        raise ValueError("Target contains infinite values")

    # Build and train the regression model.
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=config.RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    logger.info(
        "Regression model trained successfully: "
        "%d rows, %d features",
        len(X_train),
        X_train.shape[1],
    )

    # Save the trained model.
    if model_path is None:
        model_path = Path(config.MODELS_DIR) / "regression_model.joblib"
    else:
        model_path = Path(model_path)

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    except (OSError, ValueError, TypeError):
        logger.exception("Failed to save regression model")
        raise

    logger.info("Regression model saved to: %s", model_path)

    return model


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    trained_model = train_regression_model()

    print("\nRegression model trained successfully!")
    print("Model type:", type(trained_model).__name__)
    print("Number of features:", trained_model.n_features_in_)
    print(
        "Saved model:",
        Path(config.MODELS_DIR) / "regression_model.joblib",
    )
