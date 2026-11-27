
"""Tests for Step 3 - Clean."""

import logging

import pandas as pd
import pytest

import config
from Clean import clean_data


@pytest.fixture
def valid_data():
    """Create a valid DataFrame using the project configuration."""

    data = {}

    for col, (low, high) in config.NUMERIC_RANGES.items():
        data[col] = [
            low + (high - low) * 0.25,
            low + (high - low) * 0.50,
            low + (high - low) * 0.75,
            low + (high - low) * 0.60,
        ]

    for col, allowed in config.CATEGORICAL_VALUES.items():
        data[col] = [
            allowed[0],
            allowed[0],
            allowed[1],
            allowed[0],
        ]

    data[config.TARGET_COLUMN] = [70, 80, 90, 60]

    return pd.DataFrame(data)


# --------------------------------
# Tests for clean_data()
# --------------------------------

def test_clean_valid_data(valid_data):
    """Test cleaning a valid DataFrame."""

    cleaned = clean_data(valid_data)

    assert isinstance(cleaned, pd.DataFrame)
    assert not cleaned.empty
    assert len(cleaned) == len(valid_data)
    assert cleaned.isna().sum().sum() == 0


def test_clean_non_dataframe():
    """Test that non-DataFrame input raises TypeError."""

    with pytest.raises(TypeError, match="expects a pandas DataFrame"):
        clean_data([1, 2, 3])


def test_clean_empty_dataframe():
    """Test that an empty DataFrame raises ValueError."""

    with pytest.raises(ValueError, match="Cannot clean an empty DataFrame"):
        clean_data(pd.DataFrame())


def test_clean_missing_columns(valid_data):
    """Test that missing required columns raise KeyError."""

    df = valid_data.drop(columns=[config.TARGET_COLUMN])

    with pytest.raises(KeyError, match="Missing required columns"):
        clean_data(df)


def test_clean_wrong_numeric_type(valid_data):
    """Test that incorrect numeric column types raise TypeError."""

    col = list(config.NUMERIC_RANGES)[0]
    valid_data[col] = ["invalid"] * len(valid_data)

    with pytest.raises(TypeError, match=f"Column '{col}' must be numeric"):
        clean_data(valid_data)


def test_clean_missing_target_values(valid_data):
    """Test that rows with missing target values are removed."""

    valid_data.loc[0, config.TARGET_COLUMN] = float("nan")

    cleaned = clean_data(valid_data)

    assert len(cleaned) == len(valid_data) - 1
    assert cleaned[config.TARGET_COLUMN].notna().all()


def test_clean_all_target_values_missing(valid_data):
    """Test that cleaning raises ValueError when all target values are missing."""

    valid_data[config.TARGET_COLUMN] = float("nan")

    with pytest.raises(
        ValueError,
        match="No rows left after removing missing target values",
    ):
        clean_data(valid_data)


def test_clean_fills_numeric_missing_with_median(valid_data):
    """Test that missing numeric values are filled with the median."""

    col = list(config.NUMERIC_RANGES)[0]
    low, high = config.NUMERIC_RANGES[col]

    values = [
        low + (high - low) * 0.20,
        low + (high - low) * 0.40,
        low + (high - low) * 0.60,
        low + (high - low) * 0.80,
    ]

    valid_data[col] = values
    valid_data.loc[1, col] = float("nan")

    expected_median = pd.Series(
        [values[0], values[2], values[3]]
    ).median()

    cleaned = clean_data(valid_data)

    assert cleaned.loc[1, col] == expected_median
    assert cleaned[col].isna().sum() == 0


def test_clean_invalid_numeric_values_do_not_affect_median(valid_data):
    """Test that out-of-range values are removed before calculating the median."""

    col = list(config.NUMERIC_RANGES)[0]
    low, high = config.NUMERIC_RANGES[col]

    valid_data[col] = [
        low + (high - low) * 0.20,
        low + (high - low) * 0.40,
        low + (high - low) * 0.60,
        high + 1,
    ]

    valid_data.loc[0, col] = float("nan")

    expected_median = pd.Series(
        [
            low + (high - low) * 0.40,
            low + (high - low) * 0.60,
        ]
    ).median()

    cleaned = clean_data(valid_data)

    assert len(cleaned) == len(valid_data) - 1
    assert cleaned.loc[0, col] == expected_median
    assert cleaned[col].between(low, high).all()


def test_clean_numeric_column_without_usable_values(valid_data):
    """Test that an entirely missing numeric column raises ValueError."""

    col = list(config.NUMERIC_RANGES)[0]
    valid_data[col] = float("nan")

    with pytest.raises(
        ValueError,
        match=f"Cannot fill missing values in '{col}'",
    ):
        clean_data(valid_data)


def test_clean_fills_categorical_missing_with_mode(valid_data):
    """Test that missing categorical values are filled with the mode."""

    col = list(config.CATEGORICAL_VALUES)[0]
    expected_mode = valid_data[col].mode().iloc[0]

    valid_data.loc[1, col] = None

    cleaned = clean_data(valid_data)

    assert cleaned.loc[1, col] == expected_mode
    assert cleaned[col].isna().sum() == 0


def test_clean_unexpected_categories_do_not_affect_mode(valid_data):
    """Test that unexpected categories are removed before calculating the mode."""

    col = list(config.CATEGORICAL_VALUES)[0]

    allowed = config.CATEGORICAL_VALUES[col]

    if len(allowed) < 2:
        pytest.skip("At least two allowed categories are required")

    valid_data[col] = [
        allowed[0],
        allowed[1],
        allowed[1],
        "Unexpected_Value",
    ]

    valid_data.loc[0, col] = None

    cleaned = clean_data(valid_data)

    assert len(cleaned) == len(valid_data) - 1
    assert cleaned.loc[0, col] == allowed[1]
    assert cleaned[col].isin(allowed).all()


def test_clean_categorical_column_without_usable_values(valid_data):
    """Test that an entirely missing categorical column raises ValueError."""

    col = list(config.CATEGORICAL_VALUES)[0]
    valid_data[col] = None

    with pytest.raises(
        ValueError,
        match=f"Cannot fill missing values in '{col}'",
    ):
        clean_data(valid_data)


def test_clean_removes_numeric_out_of_range_rows(valid_data):
    """Test that rows with invalid numeric values are removed."""

    col = list(config.NUMERIC_RANGES)[0]
    _, high = config.NUMERIC_RANGES[col]

    valid_data.loc[0, col] = high + 1

    cleaned = clean_data(valid_data)

    assert len(cleaned) == len(valid_data) - 1
    assert cleaned[col].between(*config.NUMERIC_RANGES[col]).all()


def test_clean_all_numeric_rows_invalid(valid_data):
    """Test that cleaning raises ValueError if all rows have invalid numeric values."""

    col = list(config.NUMERIC_RANGES)[0]
    _, high = config.NUMERIC_RANGES[col]

    valid_data[col] = high + 1

    with pytest.raises(
        ValueError,
        match="No rows left after removing invalid feature values",
    ):
        clean_data(valid_data)


def test_clean_removes_unexpected_categories(valid_data):
    """Test that rows with unexpected categorical values are removed."""

    col = list(config.CATEGORICAL_VALUES)[0]
    valid_data.loc[0, col] = "Unexpected_Value"

    cleaned = clean_data(valid_data)

    assert len(cleaned) == len(valid_data) - 1
    assert cleaned[col].isin(config.CATEGORICAL_VALUES[col]).all()


def test_clean_removes_invalid_exam_scores(valid_data):
    """Test that rows with exam scores outside the valid range are removed."""

    valid_data.loc[0, config.TARGET_COLUMN] = config.SCORE_MAX + 1

    cleaned = clean_data(valid_data)

    assert len(cleaned) == len(valid_data) - 1
    assert cleaned[config.TARGET_COLUMN].between(
        config.SCORE_MIN,
        config.SCORE_MAX,
    ).all()


def test_clean_all_exam_scores_invalid(valid_data):
    """Test that cleaning raises ValueError if all exam scores are invalid."""

    valid_data[config.TARGET_COLUMN] = config.SCORE_MAX + 1

    with pytest.raises(
        ValueError,
        match="No rows left after removing invalid exam scores",
    ):
        clean_data(valid_data)


def test_clean_removes_duplicate_rows(valid_data):
    """Test that exact duplicate rows are removed."""

    df = pd.concat(
        [valid_data, valid_data.iloc[[0]]],
        ignore_index=True,
    )

    cleaned = clean_data(df)

    assert len(cleaned) == len(valid_data)
    assert cleaned.duplicated().sum() == 0


def test_clean_resets_index(valid_data):
    """Test that the cleaned DataFrame has a reset index."""

    valid_data.index = [10, 20, 30, 40]

    cleaned = clean_data(valid_data)

    assert cleaned.index.tolist() == list(range(len(cleaned)))


def test_clean_does_not_modify_original_data(valid_data):
    """Test that cleaning does not modify the original DataFrame."""

    original = valid_data.copy(deep=True)

    clean_data(valid_data)

    pd.testing.assert_frame_equal(valid_data, original)


def test_clean_logs_success(valid_data, caplog):
    """Test that cleaning generates an INFO log."""

    with caplog.at_level(logging.INFO, logger="Clean"):
        clean_data(valid_data)

    assert "Cleaning finished:" in caplog.text
