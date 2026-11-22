
import numpy as np
import pytest

from Regression_Evaluation import (
    validate_actual_scores,
    validate_predictions,
    calculate_regression_metrics
)


def test_validate_actual_scores():
    actual_scores = np.array([80, 85, 90])

    validate_actual_scores(actual_scores)


def test_validate_actual_scores_with_invalid_values():
    actual_scores = np.array([80, np.nan, 90])

    with pytest.raises(ValueError):
        validate_actual_scores(actual_scores)


def test_validate_predictions():
    actual_scores = np.array([80, 85, 90])
    predicted_scores = np.array([81, 84, 89])

    validate_predictions(
        actual_scores,
        predicted_scores,
        "Linear Regression"
    )


def test_validate_predictions_with_different_lengths():
    actual_scores = np.array([80, 85, 90])
    predicted_scores = np.array([81, 84])

    with pytest.raises(ValueError):
        validate_predictions(
            actual_scores,
            predicted_scores,
            "Linear Regression"
        )


def test_validate_predictions_with_invalid_values():
    actual_scores = np.array([80, 85, 90])
    predicted_scores = np.array([81, np.nan, 89])

    with pytest.raises(ValueError):
        validate_predictions(
            actual_scores,
            predicted_scores,
            "Linear Regression"
        )


def test_calculate_regression_metrics():
    actual_scores = np.array([3, -0.5, 2, 7])
    predicted_scores = np.array([2.5, 0, 2, 8])

    metrics = calculate_regression_metrics(
        actual_scores,
        predicted_scores
    )

    assert metrics["MAE"] == pytest.approx(0.5)
    assert metrics["MSE"] == pytest.approx(0.375)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(0.375))
    assert metrics["R2"] == pytest.approx(0.948608)
