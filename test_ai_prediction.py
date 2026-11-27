
"""Tests for the AI prediction module."""

import numpy as np
import pytest

from AI_Prediction import (
    StudentPerformancePredictor,
    validate_student_input,
)


VALID_STUDENT_DATA = {
    "Hours_Studied": 25,
    "Attendance": 90,
    "Sleep_Hours": 8,
    "Previous_Scores": 85,
    "Tutoring_Sessions": 3,
    "Physical_Activity": 5,
    "Parental_Involvement": "High",
    "Access_to_Resources": "High",
    "Extracurricular_Activities": "Yes",
    "Motivation_Level": "High",
    "Internet_Access": "Yes",
    "Family_Income": "Medium",
    "Teacher_Quality": "High",
    "School_Type": "Public",
    "Peer_Influence": "Positive",
    "Learning_Disabilities": "No",
    "Parental_Education_Level": "College",
    "Distance_from_Home": "Near",
    "Gender": "Female",
}


@pytest.fixture
def predictor():
    """Load the saved prediction models."""
    return StudentPerformancePredictor()


def test_validate_valid_student_input():
    """Valid student data should pass validation."""

    result = validate_student_input(VALID_STUDENT_DATA)

    assert len(result) == 1
    assert result.shape[1] == 19


def test_reject_missing_features():
    """Missing input features should raise an error."""

    invalid_data = VALID_STUDENT_DATA.copy()
    invalid_data.pop("Attendance")

    with pytest.raises(ValueError):
        validate_student_input(invalid_data)


def test_reject_invalid_numeric_range():
    """Numeric values outside allowed ranges should be rejected."""

    invalid_data = VALID_STUDENT_DATA.copy()
    invalid_data["Attendance"] = 150

    with pytest.raises(ValueError):
        validate_student_input(invalid_data)


def test_reject_invalid_categorical_value():
    """Invalid categorical values should be rejected."""

    invalid_data = VALID_STUDENT_DATA.copy()
    invalid_data["Gender"] = "Unknown"

    with pytest.raises(ValueError):
        validate_student_input(invalid_data)


def test_load_saved_models(predictor):
    """All saved models should load successfully."""

    assert predictor.preprocessor is not None
    assert predictor.regression_model is not None
    assert predictor.anomaly_model is not None


def test_predict_student_performance(predictor):
    """Prediction should return valid results."""

    results = predictor.predict(VALID_STUDENT_DATA)

    assert np.isfinite(results["predicted_exam_score"])
    assert results["anomaly_status"] in ("Normal", "Anomaly")
    assert results["anomaly_prediction"] in (1, -1)
    assert np.isfinite(results["anomaly_score"])


def test_prediction_score_range(predictor):
    """Predicted exam score should be within the valid score range."""

    results = predictor.predict(VALID_STUDENT_DATA)

    assert 0 <= results["predicted_exam_score"] <= 100
