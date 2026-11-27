
"""Step 6 - Transform: split the data, scale numeric features,
and encode categorical features.
"""

import logging
import math

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

logger = logging.getLogger(__name__)

# Performance levels used to preserve class proportions during splitting.
PERFORMANCE_LABELS = ["Low", "Medium", "High"]


def _get_performance_levels(target_values):
    """Assign performance levels using the project's score boundaries."""

    levels = pd.Series(index=target_values.index, dtype="object")

    levels.loc[target_values <= 60] = "Low"

    levels.loc[
        (target_values > 60) & (target_values < 80)
    ] = "Medium"

    levels.loc[target_values >= 80] = "High"

    return levels


def build_preprocessor():
    """Build the preprocessing pipeline for numeric and categorical features."""

    numeric_cols = list(config.NUMERIC_RANGES)
    categorical_cols = list(config.CATEGORICAL_VALUES)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_cols,
            ),
        ],
        verbose_feature_names_out=False,
    )


def split_features_target(df):
    """Separate input features (X) from the target (y)."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("split_features_target expects a pandas DataFrame")

    if df.empty:
        raise ValueError("Cannot split an empty DataFrame")

    feature_cols = (
        list(config.NUMERIC_RANGES)
        + list(config.CATEGORICAL_VALUES)
    )

    required_cols = feature_cols + [config.TARGET_COLUMN]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if df[required_cols].isna().any().any():
        raise ValueError(
            "Missing values found. Clean the data before transformation."
        )

    for col in list(config.NUMERIC_RANGES) + [config.TARGET_COLUMN]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric")

    X = df[feature_cols].copy()
    y = df[config.TARGET_COLUMN].copy()

    logger.info(
        "Separated features and target: %d features",
        len(feature_cols),
    )

    return X, y


def split_train_test(X, y):
    """Split the features and target using stratified performance levels."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError("y must be a pandas Series or DataFrame")

    if X.empty or y.empty:
        raise ValueError("Cannot split empty data")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")

    if not 0 < config.TEST_SIZE < 1:
        raise ValueError("TEST_SIZE must be greater than 0 and less than 1")

    if len(X) < 2:
        raise ValueError("At least two rows are required for splitting")

    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("y must contain exactly one target column")

        target_values = y.iloc[:, 0]
    else:
        target_values = y

    if not pd.api.types.is_numeric_dtype(target_values):
        raise TypeError("Target values must be numeric")

    if target_values.isna().any():
        raise ValueError("Target values must not contain missing values")

    # Derive performance levels from the original target scores.
    performance_levels = _get_performance_levels(target_values)

    class_counts = performance_levels.value_counts()
    observed_counts = class_counts[class_counts > 0]
    number_of_classes = len(observed_counts)

    # Stratification requires at least two rows in every observed class.
    if (observed_counts < 2).any():
        raise ValueError(
            "Each performance level must contain at least two rows "
            "for stratified splitting"
        )

    # Match scikit-learn's test-size calculation for float test_size.
    test_rows = math.ceil(len(X) * config.TEST_SIZE)
    train_rows = len(X) - test_rows

    if test_rows < number_of_classes or train_rows < number_of_classes:
        raise ValueError(
            "Training and testing sets must each contain at least "
            "one row per performance level"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=performance_levels,
    )

    logger.info(
        "Stratified data split completed: train=%d rows, test=%d rows",
        len(X_train),
        len(X_test),
    )

    return X_train, X_test, y_train, y_test


def transform_data(df):
    """Split the data and fit preprocessing on training data only."""

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    preprocessor = build_preprocessor()

    # Fit only on training data to prevent data leakage.
    train_array = preprocessor.fit_transform(X_train)

    # Apply the learned transformations to testing data.
    test_array = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    train_df = pd.DataFrame(
        train_array,
        columns=feature_names,
        index=X_train.index,
    )

    test_df = pd.DataFrame(
        test_array,
        columns=feature_names,
        index=X_test.index,
    )

    # Keep the original target values without scaling.
    train_df[config.TARGET_COLUMN] = y_train
    test_df[config.TARGET_COLUMN] = y_test

    logger.info(
        "Transformation completed: %d transformed features, "
        "train=%d rows, test=%d rows",
        len(feature_names),
        len(train_df),
        len(test_df),
    )

    return train_df, test_df, preprocessor


if __name__ == "__main__":
    from Clean import clean_data
    from Load import load_data

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    data = clean_data(load_data())

    train, test, preprocessor = transform_data(data)

    print("\nTrain shape:", train.shape)
    print("Test shape:", test.shape)

    print(
        "\nMissing values:",
        int(train.isna().sum().sum() + test.isna().sum().sum()),
    )

    print(
        "\nMean of Hours_Studied in train:",
        round(train["Hours_Studied"].mean(), 6),
    )

    print(
        "\nMean of Hours_Studied in test:",
        round(test["Hours_Studied"].mean(), 4),
    )

    print(
        "\nTarget column:",
        config.TARGET_COLUMN,
    )

    print(
        "Target scaling:",
        "Not applied",
    )

    print(
        "\nPreprocessor fitted:",
        hasattr(preprocessor, "feature_names_in_"),
    )
