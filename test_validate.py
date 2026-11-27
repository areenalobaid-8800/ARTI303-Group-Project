
"""Tests for Step 2 - Validate."""

import logging

import numpy as np
import pandas as pd
import pytest

import config
from Validate import validate_data


@pytest.fixture
def valid_data():
    """Create a valid DataFrame using the project configuration."""

    data = {}
    n_rows = 4

    for col, (low, high) in config.NUMERIC_RANGES.items():
        data[col] = np.linspace(
            low + (high - low) * 0.2,
            low + (high - low) * 0.8,
            n_rows,
        )

    for col, allowed in config.CATEGORICAL_VALUES.items():
        data[col] = [
            allowed[i % len(allowed)]
            for i in range(n_rows)
        ]

    data[config.TARGET_COLUMN] = [65, 75, 85, 90]

    return pd.DataFrame(data)


# --------------------------------
# Tests for validate_data()
# --------------------------------

def test_validate_valid_dataframe(valid_data):
    """Test that a valid DataFrame passes validation."""

    report = validate_data(valid_data)

    assert report["is_valid"] is True
    assert report["errors"] == []
    assert report["warnings"] == []
    assert report["missing_values"] == {}
    assert report["duplicate_rows"] == 0


def test_validate_non_dataframe():
    """Test that non-DataFrame input raises TypeError."""

    with pytest.raises(
        TypeError,
        match="validate_data expects a pandas DataFrame",
    ):
        validate_data([1, 2, 3])


def test_validate_empty_dataframe():
    """Test that an empty DataFrame raises ValueError."""

    with pytest.raises(
        ValueError,
        match="Cannot validate an empty DataFrame",
    ):
        validate_data(pd.DataFrame())


def test_validate_missing_column(valid_data):
    """Test that missing required columns are reported as errors."""

    col = list(config.NUMERIC_RANGES)[0]
    invalid_data = valid_data.drop(columns=[col])

    report = validate_data(invalid_data)

    assert report["is_valid"] is False
    assert any(
        "Missing required columns" in error
        for error in report["errors"]
    )


def test_validate_wrong_numeric_type(valid_data):
    """Test that a numeric column with the wrong type is an error."""

    col = list(config.NUMERIC_RANGES)[0]
    valid_data[col] = ["invalid"] * len(valid_data)

    report = validate_data(valid_data)

    assert report["is_valid"] is False
    assert any(
        f"Column '{col}' must be numeric" in error
        for error in report["errors"]
    )


def test_validate_missing_values_warning(valid_data):
    """Test that missing values generate a warning."""

    col = list(config.NUMERIC_RANGES)[0]
    valid_data.loc[0, col] = np.nan

    report = validate_data(valid_data)

    assert report["is_valid"] is True
    assert report["errors"] == []
    assert col in report["missing_values"]
    assert report["missing_values"][col] == 1
    assert any(
        "Missing values" in warning
        for warning in report["warnings"]
    )


def test_validate_duplicate_rows_warning(valid_data):
    """Test that duplicate rows generate a warning."""

    valid_data.loc[1] = valid_data.loc[0]

    report = validate_data(valid_data)

    assert report["is_valid"] is True
    assert report["duplicate_rows"] == 1
    assert any(
        "duplicate rows" in warning
        for warning in report["warnings"]
    )


def test_validate_numeric_range_warning(valid_data):
    """Test that numeric values outside the allowed range generate a warning."""

    col = list(config.NUMERIC_RANGES)[0]
    _, high = config.NUMERIC_RANGES[col]

    valid_data.loc[0, col] = high + 1

    report = validate_data(valid_data)

    assert report["is_valid"] is True
    assert report["errors"] == []
    assert any(
        f"Column '{col}'" in warning
        and "values outside" in warning
        for warning in report["warnings"]
    )


def test_validate_target_range_warning(valid_data):
    """Test that Exam_Score outside its valid range generates a warning."""

    valid_data.loc[0, config.TARGET_COLUMN] = (
        config.SCORE_MAX + 1
    )

    report = validate_data(valid_data)

    assert report["is_valid"] is True
    assert report["errors"] == []
    assert any(
        f"{config.TARGET_COLUMN} outside" in warning
        for warning in report["warnings"]
    )


def test_validate_unexpected_category_warning(valid_data):
    """Test that unexpected categorical values generate a warning."""

    col = list(config.CATEGORICAL_VALUES)[0]
    valid_data.loc[0, col] = "Unknown"

    report = validate_data(valid_data)

    assert report["is_valid"] is True
    assert report["errors"] == []
    assert any(
        f"Column '{col}'" in warning
        and "unexpected values" in warning
        for warning in report["warnings"]
    )


def test_validate_does_not_modify_original(valid_data):
    """Test that validation does not modify the original DataFrame."""

    original = valid_data.copy(deep=True)

    validate_data(valid_data)

    pd.testing.assert_frame_equal(valid_data, original)


def test_validate_logs_success(valid_data, caplog):
    """Test that successful validation generates an INFO log."""

    with caplog.at_level(logging.INFO, logger="Validate"):
        report = validate_data(valid_data)

    assert report["is_valid"] is True
    assert "Validation finished" in caplog.text
