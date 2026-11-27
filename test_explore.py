
"""Tests for Step 4 - Explore."""

import logging

import pandas as pd
import pytest

import config
from Explore import explore_data, top_correlations


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
        data[col] = [allowed[0], allowed[0], allowed[1], allowed[0]]

    data[config.TARGET_COLUMN] = [60, 70, 80, 90]

    return pd.DataFrame(data)


@pytest.fixture
def valid_correlations():
    """Create a correlation DataFrame for testing top_correlations."""

    target = config.TARGET_COLUMN

    return pd.DataFrame(
        {
            target: [0.90, -0.80, 0.30, 1.00],
            "Feature_A": [1.00, 0.20, 0.10, 0.90],
            "Feature_B": [0.20, 1.00, 0.40, -0.80],
            "Feature_C": [0.10, 0.40, 1.00, 0.30],
        },
        index=["Feature_A", "Feature_B", "Feature_C", target],
    )


# -------------------------
# Tests for explore_data()
# -------------------------

def test_explore_valid_data(valid_data):
    """Test exploration with a valid DataFrame."""

    results = explore_data(valid_data)

    assert isinstance(results, dict)

    expected_keys = {
        "shape",
        "numeric_summary",
        "categorical_counts",
        "correlations",
        "mean_score_by_category",
    }

    assert set(results.keys()) == expected_keys


def test_explore_non_dataframe():
    """Test that non-DataFrame input raises TypeError."""

    with pytest.raises(
        TypeError,
        match="explore_data expects a pandas DataFrame",
    ):
        explore_data([1, 2, 3])


def test_explore_empty_dataframe():
    """Test that an empty DataFrame raises ValueError."""

    with pytest.raises(
        ValueError,
        match="Cannot explore an empty DataFrame",
    ):
        explore_data(pd.DataFrame())


def test_explore_missing_columns(valid_data):
    """Test that missing required columns raise KeyError."""

    col = config.TARGET_COLUMN
    df = valid_data.drop(columns=[col])

    with pytest.raises(KeyError, match="Missing required columns"):
        explore_data(df)


def test_explore_wrong_numeric_type(valid_data):
    """Test that a non-numeric feature column raises TypeError."""

    col = list(config.NUMERIC_RANGES)[0]
    valid_data[col] = ["invalid"] * len(valid_data)

    with pytest.raises(
        TypeError,
        match=f"Column '{col}' must be numeric",
    ):
        explore_data(valid_data)


def test_explore_wrong_target_type(valid_data):
    """Test that a non-numeric target column raises TypeError."""

    target = config.TARGET_COLUMN
    valid_data[target] = ["Low", "Medium", "High", "Low"]

    with pytest.raises(
        TypeError,
        match=f"Column '{target}' must be numeric",
    ):
        explore_data(valid_data)


def test_explore_shape(valid_data):
    """Test that the returned shape matches the original DataFrame."""

    results = explore_data(valid_data)

    assert results["shape"] == valid_data.shape


def test_explore_numeric_summary(valid_data):
    """Test that numeric summary contains the expected columns."""

    results = explore_data(valid_data)

    summary = results["numeric_summary"]

    expected_columns = [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]

    assert list(summary.columns) == expected_columns

    expected_rows = list(config.NUMERIC_RANGES) + [config.TARGET_COLUMN]

    assert list(summary.index) == expected_rows
    assert (summary.loc[:, "count"] == len(valid_data)).all()


def test_explore_categorical_counts(valid_data):
    """Test that categorical counts are calculated correctly."""

    results = explore_data(valid_data)

    counts = results["categorical_counts"]

    assert set(counts.keys()) == set(config.CATEGORICAL_VALUES)

    for col in config.CATEGORICAL_VALUES:
        assert counts[col].sum() == len(valid_data)
        pd.testing.assert_series_equal(
            counts[col],
            valid_data[col].value_counts(),
        )


def test_explore_correlations(valid_data):
    """Test that the correlation matrix contains the expected columns."""

    results = explore_data(valid_data)

    correlations = results["correlations"]
    expected_columns = list(config.NUMERIC_RANGES) + [config.TARGET_COLUMN]

    assert list(correlations.columns) == expected_columns
    assert list(correlations.index) == expected_columns


def test_explore_correlation_values(valid_data):
    """Test that correlation values match pandas calculation."""

    results = explore_data(valid_data)

    numeric_cols = list(config.NUMERIC_RANGES) + [config.TARGET_COLUMN]

    expected = valid_data[numeric_cols].corr()

    pd.testing.assert_frame_equal(
        results["correlations"],
        expected,
    )


def test_explore_mean_score_by_category(valid_data):
    """Test that mean scores are calculated for each category."""

    results = explore_data(valid_data)

    means = results["mean_score_by_category"]
    target = config.TARGET_COLUMN

    assert set(means.keys()) == set(config.CATEGORICAL_VALUES)

    for col in config.CATEGORICAL_VALUES:
        expected = valid_data.groupby(col)[target].mean().sort_values()

        pd.testing.assert_series_equal(
            means[col],
            expected,
        )


def test_explore_mean_scores_sorted(valid_data):
    """Test that mean scores by category are sorted in ascending order."""

    results = explore_data(valid_data)

    for col in config.CATEGORICAL_VALUES:
        values = results["mean_score_by_category"][col]

        assert values.is_monotonic_increasing


def test_explore_does_not_modify_original(valid_data):
    """Test that exploration does not modify the original DataFrame."""

    original = valid_data.copy(deep=True)

    explore_data(valid_data)

    pd.testing.assert_frame_equal(valid_data, original)


def test_explore_logs_success(valid_data, caplog):
    """Test that exploration generates an INFO log."""

    with caplog.at_level(logging.INFO, logger="Explore"):
        explore_data(valid_data)

    assert (
        f"Exploration finished: {len(valid_data)} rows, "
        f"{len(valid_data.columns)} columns"
    ) in caplog.text


# ------------------------------
# Tests for top_correlations()
# ------------------------------

def test_top_correlations_valid_data(valid_correlations):
    """Test that top correlations are returned in the correct order."""

    result = top_correlations(valid_correlations, n=3)

    target = config.TARGET_COLUMN

    assert isinstance(result, pd.Series)
    assert list(result.index) == [
        "Feature_A",
        "Feature_B",
        "Feature_C",
    ]
    assert result.name == target


def test_top_correlations_default_n(valid_correlations):
    """Test that the default value of n returns five or fewer features."""

    result = top_correlations(valid_correlations)

    assert isinstance(result, pd.Series)
    assert len(result) <= 5
    assert config.TARGET_COLUMN not in result.index


def test_top_correlations_orders_by_absolute_value(valid_correlations):
    """Test that negative correlations are ranked by absolute value."""

    result = top_correlations(valid_correlations, n=3)

    assert abs(result.iloc[0]) >= abs(result.iloc[1])
    assert abs(result.iloc[1]) >= abs(result.iloc[2])

    assert result.index.tolist() == [
        "Feature_A",
        "Feature_B",
        "Feature_C",
    ]


def test_top_correlations_zero_n(valid_correlations):
    """Test that n=0 returns an empty Series."""

    result = top_correlations(valid_correlations, n=0)

    assert isinstance(result, pd.Series)
    assert result.empty


def test_top_correlations_n_larger_than_features(valid_correlations):
    """Test that requesting more features than available returns all features."""

    result = top_correlations(valid_correlations, n=10)

    assert len(result) == 3


@pytest.mark.parametrize("invalid_n", [-1, 1.5, "3", None, True, False])
def test_top_correlations_invalid_n(valid_correlations, invalid_n):
    """Test that invalid n values raise TypeError."""

    with pytest.raises(TypeError, match="n must be a non-negative integer"):
        top_correlations(valid_correlations, n=invalid_n)


def test_top_correlations_non_dataframe():
    """Test that non-DataFrame input raises TypeError."""

    with pytest.raises(
        TypeError,
        match="correlations must be a pandas DataFrame",
    ):
        top_correlations([[1, 0.5], [0.5, 1]])


def test_top_correlations_missing_target_column():
    """Test that a missing target column raises KeyError."""

    correlations = pd.DataFrame(
        {"Feature_A": [1.0, 0.5]},
        index=["Feature_A", "Feature_B"],
    )

    with pytest.raises(
        KeyError,
        match="Target column",
    ):
        top_correlations(correlations)


def test_top_correlations_missing_target_row():
    """Test that a missing target row raises KeyError."""

    target = config.TARGET_COLUMN

    correlations = pd.DataFrame(
        {target: [0.5]},
        index=["Feature_A"],
    )

    with pytest.raises(
        KeyError,
        match="Target row",
    ):
        top_correlations(correlations)
