

import numpy as np
import pytest

from Classification_evaluation import (
    create_performance_labels,
    validate_actual_labels,
    validate_predictions,
    calculate_classification_metrics,
    analyze_classification_errors
)


# ==================================================
# 1. Test Performance Labels
# ==================================================

def test_create_performance_labels():

    scores = [50, 60, 75, 80, 95]

    labels = create_performance_labels(scores)

    expected = np.array([
        "Low",
        "Medium",
        "Medium",
        "High",
        "High"
    ])

    assert np.array_equal(labels, expected)


def test_create_performance_labels_with_invalid_values():

    scores = [50, np.nan, 80]

    with pytest.raises(ValueError):
        create_performance_labels(scores)


def test_create_performance_labels_out_of_range():

    scores = [50, 105, 80]

    with pytest.raises(ValueError):
        create_performance_labels(scores)


# ==================================================
# 2. Test Actual Labels Validation
# ==================================================

def test_validate_actual_labels():

    actual_labels = np.array([
        "Low", "Medium", "High"
    ])

    validate_actual_labels(actual_labels)


def test_validate_actual_labels_with_invalid_class():

    actual_labels = np.array([
        "Low", "Medium", "Excellent"
    ])

    with pytest.raises(ValueError):
        validate_actual_labels(actual_labels)


# ==================================================
# 3. Test Predictions Validation
# ==================================================

def test_validate_predictions():

    actual = np.array([
        "Low", "Medium", "High"
    ])

    predicted = np.array([
        "Low", "Medium", "High"
    ])

    validate_predictions(
        actual,
        predicted,
        "Decision Tree"
    )


def test_validate_predictions_with_different_lengths():

    actual = np.array([
        "Low", "Medium", "High"
    ])

    predicted = np.array([
        "Low", "High"
    ])

    with pytest.raises(ValueError):
        validate_predictions(
            actual,
            predicted,
            "Decision Tree"
        )


def test_validate_predictions_with_invalid_class():

    actual = np.array([
        "Low", "Medium", "High"
    ])

    predicted = np.array([
        "Low", "Excellent", "High"
    ])

    with pytest.raises(ValueError):
        validate_predictions(
            actual,
            predicted,
            "Decision Tree"
        )


# ==================================================
# 4. Test Classification Metrics
# ==================================================

def test_calculate_classification_metrics():

    actual = np.array([
        "Low", "Medium", "High", "Low"
    ])

    predicted = np.array([
        "Low", "Medium", "High", "Medium"
    ])

    metrics = calculate_classification_metrics(
        actual,
        predicted
    )

    assert metrics["Accuracy"] == pytest.approx(0.75)

    assert metrics["Precision"] == pytest.approx(
        (1.0 + 0.5 + 1.0) / 3
    )

    assert metrics["Recall"] == pytest.approx(
        (0.5 + 1.0 + 1.0) / 3
    )

    assert metrics["F1-Score"] == pytest.approx(
        (2/3 + 2/3 + 1.0) / 3
    )


# ==================================================
# 5. Test Classification Error Analysis
# ==================================================

def test_analyze_classification_errors():

    actual = np.array([
        "Low", "Medium", "High", "High"
    ])

    predicted = np.array([
        "Low", "High", "Medium", "High"
    ])

    errors = analyze_classification_errors(
        actual,
        predicted
    )

    assert len(errors) == 2

    assert errors["Count"].sum() == 2


def test_analyze_classification_errors_without_errors():

    actual = np.array([
        "Low", "Medium", "High"
    ])

    predicted = np.array([
        "Low", "Medium", "High"
    ])

    errors = analyze_classification_errors(
        actual,
        predicted
    )

    assert errors.empty
