
"""Anomaly Detection: train an Isolation Forest model."""

import logging

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest

import config

logger = logging.getLogger(__name__)


def prepare_anomaly_features(train_df):
    """Validate the training data and exclude the target column."""

    if not isinstance(train_df, pd.DataFrame):
        raise TypeError("train_df must be a pandas DataFrame")

    if train_df.empty:
        raise ValueError("Cannot train anomaly detection on empty data")

    if train_df.columns.duplicated().any():
        raise ValueError("Training data contains duplicate columns")

    if config.TARGET_COLUMN not in train_df.columns:
        raise KeyError(
            f"Missing target column: {config.TARGET_COLUMN}"
        )

    # Exclude Exam_Score to prevent target leakage.
    X_train = train_df.drop(columns=[config.TARGET_COLUMN]).copy()

    if X_train.empty:
        raise ValueError("No features available for anomaly detection")

    if len(X_train) < 2:
        raise ValueError("At least two rows are required for training")

    if X_train.isna().any().any():
        raise ValueError("Missing values found in training features")

    for col in X_train.columns:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            raise TypeError(
                f"Feature '{col}' must contain numeric values"
            )

    values = X_train.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Training features must contain only finite values"
        )

    logger.info(
        "Anomaly detection features prepared: %d rows, %d features",
        len(X_train),
        X_train.shape[1],
    )

    return X_train


def train_anomaly_model(
    train_df=None,
    model_path=None,
    contamination=0.05,
):
    """Train and save an Isolation Forest model."""

    if not isinstance(contamination, (int, float)):
        raise TypeError("contamination must be numeric")

    if not 0 < contamination <= 0.5:
        raise ValueError(
            "contamination must be greater than 0 and at most 0.5"
        )

    if train_df is None:
        train_path = config.PROCESSED_DATA_DIR / "train.csv"

        try:
            train_df = pd.read_csv(train_path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.exception("Failed to load training data")
            raise ValueError(
                f"Could not load training data: {train_path}"
            ) from exc

    X_train = prepare_anomaly_features(train_df)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    try:
        model.fit(X_train)

        # Isolation Forest returns:
        # 1  = normal observation
        # -1 = anomaly
        predictions = model.predict(X_train)

        # Lower scores indicate more unusual observations.
        anomaly_scores = model.decision_function(X_train)

        if not np.isfinite(anomaly_scores).all():
            raise ValueError(
                "Model produced non-finite anomaly scores"
            )

        if model_path is None:
            model_path = (
                config.MODELS_DIR / "anomaly_detection_model.joblib"
            )

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    except (OSError, ValueError, TypeError):
        logger.exception("Anomaly detection training or saving failed")
        raise

    anomaly_count = int((predictions == -1).sum())
    normal_count = int((predictions == 1).sum())

    logger.info(
        "Isolation Forest trained successfully: %d rows, %d features",
        len(X_train),
        X_train.shape[1],
    )

    logger.info(
        "Training results: %d normal observations, %d anomalies",
        normal_count,
        anomaly_count,
    )

    logger.info(
        "Anomaly detection model saved to: %s",
        model_path,
    )

    return {
        "model": model,
        "predictions": predictions,
        "anomaly_scores": anomaly_scores,
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "contamination": contamination,
        "model_path": model_path,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    result = train_anomaly_model()

    print("\nAnomaly Detection training completed.")
    print("Model type:", type(result["model"]).__name__)
    print("Number of training rows:", len(result["predictions"]))
    print("Number of normal observations:", result["normal_count"])
    print("Number of anomalies:", result["anomaly_count"])
    print("Contamination setting:", result["contamination"])
    print("Model saved to:", result["model_path"])

    print(
        "\nAnomaly percentage:",
        round(
            result["anomaly_count"] / len(result["predictions"]) * 100,
            2,
        ),
        "%",
    )

    print(
        "Mean anomaly score:",
        round(float(np.mean(result["anomaly_scores"])), 4),
    )
