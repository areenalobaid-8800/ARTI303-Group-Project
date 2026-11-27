
"""Tests for Step 6 - Transform."""

import logging

import numpy as np
import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
import Transform


@pytest.fixture
def valid_data():
    """Create a valid DataFrame using the project configuration."""

    data = {}
    n_rows = 20

    for col, (low, high) in config.NUMERIC_RANGES.items():
        data[col] = np.linspace(
            low + (high - low) * 0.1,
            low + (high - low) * 0.9,
            n_rows,
        )

    for col, allowed in config.CATEGORICAL_VALUES.items():
        data[col] = pd.Series(
            [allowed[i % len(allowed)] for i in range(n_rows)],
            dtype=object,
        )

    data[config.TARGET_COLUMN] = [
        50, 55, 58, 60,
        61, 63, 65, 67, 69, 71,
        73, 75, 77, 79,
        80, 82, 84, 86, 88, 90,
    ]

    return pd.DataFrame(data)


# --------------------------------
# Tests for build_preprocessor()
# --------------------------------

def test_build_preprocessor_returns_column_transformer():
    """Test that the preprocessor is a ColumnTransformer."""

    preprocessor = Transform.build_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)


def test_build_preprocessor_numeric_columns():
    """Test that numeric columns use StandardScaler."""

    preprocessor = Transform.build_preprocessor()

    numeric_transformer = preprocessor.transformers[0][1]
    numeric_columns = preprocessor.transformers[0][2]

    assert isinstance(numeric_transformer, StandardScaler)
    assert list(numeric_columns) == list(config.NUMERIC_RANGES)


def test_build_preprocessor_categorical_columns():
    """Test that categorical columns use OneHotEncoder."""

    preprocessor = Transform.build_preprocessor()

    categorical_transformer = preprocessor.transformers[1][1]
    categorical_columns = preprocessor.transformers[1][2]

    assert isinstance(categorical_transformer, OneHotEncoder)
    assert categorical_transformer.handle_unknown == "ignore"
    assert categorical_transformer.sparse_output is False

    assert list(categorical_columns) == list(config.CATEGORICAL_VALUES)


# ------------------------------------
# Tests for performance level assignment
# ------------------------------------

def test_get_performance_levels_boundaries():
    """Test that scores are assigned to the correct performance levels."""

    scores = pd.Series(
        [0, 50, 60, 60.1, 70, 79.9, 80, 90, 100]
    )

    levels = Transform._get_performance_levels(scores)

    expected = pd.Series(
        [
            "Low",
            "Low",
            "Low",
            "Medium",
            "Medium",
            "Medium",
            "High",
            "High",
            "High",
        ],
        dtype="object",
    )

    pd.testing.assert_series_equal(
        levels.reset_index(drop=True),
        expected,
    )


# ------------------------------------
# Tests for split_features_target()
# ------------------------------------

def test_split_features_target_returns_X_and_y(valid_data):
    """Test that the function separates features and target correctly."""

    X, y = Transform.split_features_target(valid_data)

    expected_features = (
        list(config.NUMERIC_RANGES)
        + list(config.CATEGORICAL_VALUES)
    )

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    assert list(X.columns) == expected_features
    assert y.name == config.TARGET_COLUMN

    assert len(X) == len(valid_data)
    assert len(y) == len(valid_data)


def test_split_features_target_feature_count(valid_data):
    """Test that X contains all expected input features."""

    X, y = Transform.split_features_target(valid_data)

    expected_count = (
        len(config.NUMERIC_RANGES)
        + len(config.CATEGORICAL_VALUES)
    )

    assert X.shape[1] == expected_count
    assert config.TARGET_COLUMN not in X.columns


def test_split_features_target_target_values(valid_data):
    """Test that the original target values are preserved."""

    X, y = Transform.split_features_target(valid_data)

    pd.testing.assert_series_equal(
        y,
        valid_data[config.TARGET_COLUMN],
    )


def test_split_features_target_non_dataframe():
    """Test that non-DataFrame input raises TypeError."""

    with pytest.raises(
        TypeError,
        match="split_features_target expects a pandas DataFrame",
    ):
        Transform.split_features_target([1, 2, 3])


def test_split_features_target_empty_dataframe():
    """Test that an empty DataFrame raises ValueError."""

    with pytest.raises(
        ValueError,
        match="Cannot split an empty DataFrame",
    ):
        Transform.split_features_target(pd.DataFrame())


def test_split_features_target_missing_column(valid_data):
    """Test that missing required columns raise KeyError."""

    col = list(config.NUMERIC_RANGES)[0]
    invalid_data = valid_data.drop(columns=[col])

    with pytest.raises(
        KeyError,
        match="Missing required columns",
    ):
        Transform.split_features_target(invalid_data)


def test_split_features_target_missing_values(valid_data):
    """Test that missing values raise ValueError."""

    col = list(config.NUMERIC_RANGES)[0]
    valid_data.loc[0, col] = np.nan

    with pytest.raises(
        ValueError,
        match="Missing values found",
    ):
        Transform.split_features_target(valid_data)


@pytest.mark.parametrize(
    "column",
    list(config.NUMERIC_RANGES) + [config.TARGET_COLUMN],
)
def test_split_features_target_wrong_numeric_type(valid_data, column):
    """Test that numeric columns with non-numeric types raise TypeError."""

    valid_data[column] = ["invalid"] * len(valid_data)

    with pytest.raises(
        TypeError,
        match=f"Column '{column}' must be numeric",
    ):
        Transform.split_features_target(valid_data)


def test_split_features_target_does_not_modify_original(valid_data):
    """Test that the original DataFrame is not modified."""

    original = valid_data.copy(deep=True)

    Transform.split_features_target(valid_data)

    pd.testing.assert_frame_equal(valid_data, original)


def test_split_features_target_logs_success(valid_data, caplog):
    """Test that successful separation generates an INFO log."""

    with caplog.at_level(logging.INFO, logger="Transform"):
        X, y = Transform.split_features_target(valid_data)

    assert X is not None
    assert y is not None
    assert "Separated features and target" in caplog.text


# --------------------------------
# Tests for split_train_test()
# --------------------------------

def test_split_train_test_returns_four_parts(valid_data):
    """Test that the function returns four split datasets."""

    X, y = Transform.split_features_target(valid_data)

    result = Transform.split_train_test(X, y)

    assert len(result) == 4

    X_train, X_test, y_train, y_test = result

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)


def test_split_train_test_correct_sizes(valid_data):
    """Test that train and test sizes follow TEST_SIZE."""

    X, y = Transform.split_features_target(valid_data)

    X_train, X_test, y_train, y_test = Transform.split_train_test(X, y)

    expected_test_size = int(np.ceil(len(X) * config.TEST_SIZE))

    assert len(X_test) == expected_test_size
    assert len(X_train) == len(X) - expected_test_size

    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_split_train_test_same_indices(valid_data):
    """Test that features and targets retain matching indices."""

    X, y = Transform.split_features_target(valid_data)

    X_train, X_test, y_train, y_test = Transform.split_train_test(X, y)

    assert list(X_train.index) == list(y_train.index)
    assert list(X_test.index) == list(y_test.index)


def test_split_train_test_no_overlap(valid_data):
    """Test that training and testing indices do not overlap."""

    X, y = Transform.split_features_target(valid_data)

    X_train, X_test, y_train, y_test = Transform.split_train_test(X, y)

    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_split_train_test_deterministic(valid_data):
    """Test that the split is reproducible with the configured random state."""

    X, y = Transform.split_features_target(valid_data)

    first = Transform.split_train_test(X, y)
    second = Transform.split_train_test(X, y)

    for first_part, second_part in zip(first, second):

        if isinstance(first_part, pd.DataFrame):
            pd.testing.assert_frame_equal(first_part, second_part)

        elif isinstance(first_part, pd.Series):
            pd.testing.assert_series_equal(first_part, second_part)


def test_split_train_test_preserves_all_performance_levels(valid_data):
    """Test that every observed performance level appears in train and test."""

    X, y = Transform.split_features_target(valid_data)

    X_train, X_test, y_train, y_test = Transform.split_train_test(X, y)

    all_levels = set(
        Transform._get_performance_levels(y).dropna().astype(str)
    )

    train_levels = set(
        Transform._get_performance_levels(y_train).dropna().astype(str)
    )

    test_levels = set(
        Transform._get_performance_levels(y_test).dropna().astype(str)
    )

    assert train_levels == all_levels
    assert test_levels == all_levels


def test_split_train_test_preserves_class_distribution(valid_data):
    """Test that stratification distributes each class across both sets."""

    X, y = Transform.split_features_target(valid_data)

    X_train, X_test, y_train, y_test = Transform.split_train_test(X, y)

    train_counts = (
        Transform._get_performance_levels(y_train).value_counts()
    )

    test_counts = (
        Transform._get_performance_levels(y_test).value_counts()
    )

    # The fixture has 4 Low, 10 Medium and 6 High rows.
    # With 20% testing, the expected distribution is 1, 2 and 1.
    assert test_counts["Low"] == 1
    assert test_counts["Medium"] == 2
    assert test_counts["High"] == 1

    assert train_counts["Low"] == 3
    assert train_counts["Medium"] == 8
    assert train_counts["High"] == 5


def test_split_train_test_rejects_class_with_one_row(valid_data):
    """Test that stratification rejects a performance level with one row."""

    X, y = Transform.split_features_target(valid_data)

    y = y.copy()
    y.iloc[:3] = 65

    with pytest.raises(
        ValueError,
        match="Each performance level must contain at least two rows",
    ):
        Transform.split_train_test(X, y)


def test_split_train_test_rejects_missing_target_values(valid_data):
    """Test that missing target values raise ValueError."""

    X, y = Transform.split_features_target(valid_data)

    y = y.copy()
    y.iloc[0] = np.nan

    with pytest.raises(
        ValueError,
        match="Target values must not contain missing values",
    ):
        Transform.split_train_test(X, y)


def test_split_train_test_rejects_non_numeric_target(valid_data):
    """Test that non-numeric target values raise TypeError."""

    X, y = Transform.split_features_target(valid_data)

    y = pd.Series(
        ["invalid"] * len(y),
        index=y.index,
    )

    with pytest.raises(
        TypeError,
        match="Target values must be numeric",
    ):
        Transform.split_train_test(X, y)


def test_split_train_test_rejects_multiple_target_columns(valid_data):
    """Test that a DataFrame with multiple target columns raises ValueError."""

    X, y = Transform.split_features_target(valid_data)

    multiple_targets = pd.DataFrame(
        {
            "target_1": y,
            "target_2": y,
        }
    )

    with pytest.raises(
        ValueError,
        match="y must contain exactly one target column",
    ):
        Transform.split_train_test(X, multiple_targets)


def test_split_train_test_non_dataframe_X():
    """Test that invalid X raises TypeError."""

    with pytest.raises(
        TypeError,
        match="X must be a pandas DataFrame",
    ):
        Transform.split_train_test(
            [1, 2, 3],
            pd.Series([1, 2, 3]),
        )


def test_split_train_test_invalid_y(valid_data):
    """Test that invalid y raises TypeError."""

    X, y = Transform.split_features_target(valid_data)

    with pytest.raises(
        TypeError,
        match="y must be a pandas Series or DataFrame",
    ):
        Transform.split_train_test(X, [1, 2, 3])


def test_split_train_test_empty_X():
    """Test that empty X raises ValueError."""

    with pytest.raises(
        ValueError,
        match="Cannot split empty data",
    ):
        Transform.split_train_test(
            pd.DataFrame(),
            pd.Series(dtype=float),
        )


def test_split_train_test_empty_y(valid_data):
    """Test that empty y raises ValueError."""

    X, y = Transform.split_features_target(valid_data)

    with pytest.raises(
        ValueError,
        match="Cannot split empty data",
    ):
        Transform.split_train_test(X, y.iloc[:0])


def test_split_train_test_different_lengths(valid_data):
    """Test that different X and y lengths raise ValueError."""

    X, y = Transform.split_features_target(valid_data)

    with pytest.raises(
        ValueError,
        match="X and y must contain the same number of rows",
    ):
        Transform.split_train_test(X, y.iloc[:-1])


@pytest.mark.parametrize(
    "invalid_size",
    [0, 1, -0.1, 1.1],
)
def test_split_train_test_invalid_test_size(
    valid_data,
    monkeypatch,
    invalid_size,
):
    """Test that invalid TEST_SIZE values raise ValueError."""

    X, y = Transform.split_features_target(valid_data)

    monkeypatch.setattr(config, "TEST_SIZE", invalid_size)

    with pytest.raises(
        ValueError,
        match="TEST_SIZE must be greater than 0 and less than 1",
    ):
        Transform.split_train_test(X, y)


def test_split_train_test_less_than_two_rows(valid_data):
    """Test that fewer than two rows raise ValueError."""

    X, y = Transform.split_features_target(valid_data)

    X = X.iloc[:1]
    y = y.iloc[:1]

    with pytest.raises(
        ValueError,
        match="At least two rows are required for splitting",
    ):
        Transform.split_train_test(X, y)


def test_split_train_test_logs_success(valid_data, caplog):
    """Test that a successful split generates an INFO log."""

    X, y = Transform.split_features_target(valid_data)

    with caplog.at_level(logging.INFO, logger="Transform"):
        result = Transform.split_train_test(X, y)

    assert len(result) == 4
    assert "Stratified data split completed" in caplog.text


# ----------------------------
# Tests for transform_data()
# ----------------------------

def test_transform_data_returns_three_values(valid_data):
    """Test that transformation returns train, test and preprocessor."""

    train_df, test_df, preprocessor = Transform.transform_data(valid_data)

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(preprocessor, ColumnTransformer)


def test_transform_data_correct_row_counts(valid_data):
    """Test that transformed data preserves train/test row counts."""

    train_df, test_df, preprocessor = Transform.transform_data(valid_data)

    expected_test_size = int(np.ceil(len(valid_data) * config.TEST_SIZE))

    assert len(train_df) == len(valid_data) - expected_test_size
    assert len(test_df) == expected_test_size


def test_transform_data_contains_target(valid_data):
    """Test that the original target column is included in both datasets."""

    train_df, test_df, preprocessor = Transform.transform_data(valid_data)

    target = config.TARGET_COLUMN

    assert target in train_df.columns
    assert target in test_df.columns

    assert train_df[target].between(
        config.SCORE_MIN,
        config.SCORE_MAX,
    ).all()

    assert test_df[target].between(
        config.SCORE_MIN,
        config.SCORE_MAX,
    ).all()


def test_transform_data_target_is_not_scaled(valid_data):
    """Test that target values remain unchanged after transformation."""

    train_df, test_df, preprocessor = Transform.transform_data(valid_data)

    X, y = Transform.split_features_target(valid_data)

    X_train, X_test, y_train, y_test = Transform.split_train_test(X, y)

    pd.testing.assert_series_equal(
        train_df[config.TARGET_COLUMN],
        y_train,
        check_names=False,
    )

    pd.testing.assert_series_equal(
        test_df[config.TARGET_COLUMN],
        y_test,
        check_names=False,
    )


def test_transform_data_numeric_features_are_scaled(valid_data):
    """Test that numeric training features are standardized."""

    train_df, test_df, preprocessor = Transform.transform_data(valid_data)

    for col in config.NUMERIC_RANGES:
        assert abs(train_df[col].mean()) < 1e-9
        assert train_df[col].std(ddof=0) == pytest.approx(1.0)


def test_transform_data_categorical_features_are_encoded(valid_data):
    """Test that categorical features are one-hot encoded."""

    train_df, test_df, preprocessor = Transform.transform_data(valid_data)

    categorical_cols = list(config.CATEGORICAL_VALUES)

    for col in categorical_cols:
        assert col not in train_df.columns
        assert col not in test_df.columns

    encoded_columns = [
        col
        for col in train_df.columns
        if col != config.TARGET_COLUMN
        and col not in config.NUMERIC_RANGES
    ]

    assert len(encoded_columns) > 0


def test_transform_data_has_no_missing_values(valid_data):
    """Test that transformed train and test data contain no missing values."""

    train_df, test_df, preprocessor = Transform.transform_data(valid_data)

    assert not train_df.isna().any().any()
    assert not test_df.isna().any().any()


def test_transform_data_preprocessor_is_fitted(valid_data):
    """Test that the preprocessor is fitted after transformation."""

    train_df, test_df, preprocessor = Transform.transform_data(valid_data)

    assert hasattr(preprocessor, "feature_names_in_")
    assert hasattr(preprocessor, "transformers_")
    assert hasattr(preprocessor, "get_feature_names_out")


def test_transform_data_fits_only_on_training_data(valid_data):
    """Test that the scaler learns its values from training data only."""

    train_df, test_df, preprocessor = Transform.transform_data(valid_data)

    numeric_cols = list(config.NUMERIC_RANGES)

    scaler = preprocessor.named_transformers_["num"]

    expected_means = valid_data.loc[
        train_df.index,
        numeric_cols,
    ].mean().to_numpy()

    np.testing.assert_allclose(
        scaler.mean_,
        expected_means,
    )


def test_transform_data_preserves_original_indices(valid_data):
    """Test that transformed data keeps the original row indices."""

    train_df, test_df, preprocessor = Transform.transform_data(valid_data)

    assert set(train_df.index).isdisjoint(set(test_df.index))

    combined_indices = list(train_df.index) + list(test_df.index)

    assert sorted(combined_indices) == sorted(valid_data.index.tolist())


def test_transform_data_invalid_input():
    """Test that invalid input raises TypeError."""

    with pytest.raises(
        TypeError,
        match="split_features_target expects a pandas DataFrame",
    ):
        Transform.transform_data([1, 2, 3])


def test_transform_data_logs_success(valid_data, caplog):
    """Test that successful transformation generates an INFO log."""

    with caplog.at_level(logging.INFO, logger="Transform"):
        train_df, test_df, preprocessor = Transform.transform_data(valid_data)

    assert "Transformation completed" in caplog.text
