
"""Evaluate Isolation Forest anomaly detection results."""

import logging

import joblib
import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def prepare_evaluation_features(test_df, model):
    """Validate test data and prepare features in the training order."""

    if not isinstance(test_df, pd.DataFrame):
        raise TypeError("test_df must be a pandas DataFrame")

    if test_df.empty:
        raise ValueError("Cannot evaluate an empty DataFrame")

    if test_df.columns.duplicated().any():
        raise ValueError("Test data contains duplicate columns")

    if config.TARGET_COLUMN not in test_df.columns:
        raise KeyError(
            f"Missing target column: {config.TARGET_COLUMN}"
        )

    X_test = test_df.drop(columns=[config.TARGET_COLUMN]).copy()

    if X_test.empty:
        raise ValueError("No features available for evaluation")

    if X_test.isna().any().any():
        raise ValueError("Missing values found in test features")

    for col in X_test.columns:
        if not pd.api.types.is_numeric_dtype(X_test[col]):
            raise TypeError(
                f"Feature '{col}' must contain numeric values"
            )

    values = X_test.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Test features must contain only finite values"
        )

    expected_features = list(model.feature_names_in_)
    actual_features = list(X_test.columns)

    if actual_features != expected_features:
        raise ValueError(
            "Test feature names or order do not match the trained model"
        )

    return X_test


def evaluate_anomaly_model(test_df=None, model_path=None):
    """Detect anomalies in test data and summarize their characteristics."""

    if test_df is None:
        test_path = config.PROCESSED_DATA_DIR / "test.csv"

        try:
            test_df = pd.read_csv(test_path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.exception("Failed to load testing data")
            raise ValueError(
                f"Could not load testing data: {test_path}"
            ) from exc

    if model_path is None:
        model_path = (
            config.MODELS_DIR / "anomaly_detection_model.joblib"
        )

    try:
        model = joblib.load(model_path)
    except (OSError, ValueError, TypeError) as exc:
        logger.exception("Failed to load anomaly detection model")
        raise ValueError(
            f"Could not load model: {model_path}"
        ) from exc

    X_test = prepare_evaluation_features(test_df, model)

    # Validate the target for post-detection analysis.
    target_values = test_df[config.TARGET_COLUMN]

    if not pd.api.types.is_numeric_dtype(target_values):
        raise TypeError(
            f"Target column '{config.TARGET_COLUMN}' must be numeric"
        )

    if target_values.isna().any():
        raise ValueError(
            f"Missing values found in target column: {config.TARGET_COLUMN}"
        )

    if not np.isfinite(
        target_values.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"Target column '{config.TARGET_COLUMN}' must contain only finite values"
        )

    # Detect anomalies without using the target column.
    predictions = model.predict(X_test)
    anomaly_scores = model.decision_function(X_test)

    if not np.isfinite(anomaly_scores).all():
        raise ValueError("Model produced non-finite anomaly scores")

    # Keep original scores for post-detection analysis only.
    results_df = pd.DataFrame(
        {
            "Prediction": predictions,
            "Anomaly_Score": anomaly_scores,
            config.TARGET_COLUMN: target_values.to_numpy(),
        },
        index=test_df.index,
    )

    results_df["Status"] = results_df["Prediction"].map(
        {1: "Normal", -1: "Anomaly"}
    )

    anomaly_count = int((predictions == -1).sum())
    normal_count = int((predictions == 1).sum())

    anomaly_percentage = (
        anomaly_count / len(results_df)
    ) * 100

    # Compare feature averages after detection.
    analysis_df = X_test.copy()
    analysis_df["Status"] = results_df["Status"]

    feature_means = analysis_df.groupby("Status").mean().T

    if (
        "Anomaly" in feature_means.columns
        and "Normal" in feature_means.columns
    ):
        feature_comparison = feature_means.copy()

        feature_comparison["Absolute_Difference"] = (
            feature_comparison["Anomaly"]
            - feature_comparison["Normal"]
        ).abs()

        feature_comparison = feature_comparison.sort_values(
            "Absolute_Difference",
            ascending=False,
        )
    else:
        feature_comparison = feature_means.copy()
        feature_comparison["Absolute_Difference"] = np.nan

    # Compare Exam_Score only after anomaly detection.
    score_summary = results_df.groupby("Status")[
        config.TARGET_COLUMN
    ].agg(["count", "mean", "median", "min", "max"])

    score_summary = score_summary.round(3)

    logger.info(
        "Anomaly evaluation completed: %d test rows",
        len(results_df),
    )

    logger.info(
        "Test results: %d normal observations, %d anomalies (%.2f%%)",
        normal_count,
        anomaly_count,
        anomaly_percentage,
    )

    return {
        "results": results_df,
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "anomaly_percentage": anomaly_percentage,
        "mean_anomaly_score": float(np.mean(anomaly_scores)),
        "score_summary": score_summary,
        "feature_comparison": feature_comparison,
    }


def print_results(result):
    """Print the anomaly evaluation summary."""

    print("\nAnomaly Detection Evaluation")
    print("=" * 45)

    print("Test rows:", len(result["results"]))
    print("Normal observations:", result["normal_count"])
    print("Anomalies:", result["anomaly_count"])
    print(
        "Anomaly percentage:",
        round(result["anomaly_percentage"], 2),
        "%",
    )
    print(
        "Mean anomaly score:",
        round(result["mean_anomaly_score"], 4),
    )

    print("\nExam Score Summary by Status:")
    print(result["score_summary"].to_string())

    print("\nLargest Feature Differences:")
    print(
        result["feature_comparison"]
        .head(10)
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    result = evaluate_anomaly_model()
    print_results(result)
