
"""Step 3 - Clean: fill missing values and remove invalid rows."""

import logging

import pandas as pd

import config

logger = logging.getLogger(__name__)


def clean_data(df):
    """Return a cleaned COPY of the DataFrame.

    Steps:
      1. Check the input DataFrame and required columns.
      2. Check numeric column data types.
      3. Remove rows with missing target values.
      4. Remove rows with invalid target scores.
      5. Remove rows with invalid numeric feature values.
      6. Remove rows with unexpected categorical values.
      7. Fill missing numeric values with the median.
      8. Fill missing categorical values with the mode.
      9. Remove exact duplicate rows.
      10. Reset the index.

    Missing feature values are preserved during invalid-value filtering
    so they can be filled in the next steps.

    The original DataFrame is never modified.

    Raises:
        TypeError: if the input is not a DataFrame or numeric columns
                   have incorrect data types.
        ValueError: if the DataFrame is empty or no usable values remain.
        KeyError: if required project columns are missing.
    """

    # 1. Check input
    if not isinstance(df, pd.DataFrame):
        raise TypeError("clean_data expects a pandas DataFrame")

    if df.empty:
        raise ValueError("Cannot clean an empty DataFrame")

    # 2. Check required columns
    required = (
        list(config.NUMERIC_RANGES)
        + list(config.CATEGORICAL_VALUES)
        + [config.TARGET_COLUMN]
    )

    missing_cols = [col for col in required if col not in df.columns]

    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    # 3. Check numeric column types
    for col in list(config.NUMERIC_RANGES) + [config.TARGET_COLUMN]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric")

    cleaned = df.copy()
    rows_before = len(cleaned)

    # 4. Remove rows with missing target values
    missing_target = int(cleaned[config.TARGET_COLUMN].isna().sum())

    if missing_target:
        cleaned = cleaned.dropna(subset=[config.TARGET_COLUMN])

        logger.info(
            "Dropped %d rows with missing target",
            missing_target,
        )

    if cleaned.empty:
        raise ValueError("No rows left after removing missing target values")

    # 5. Remove rows with exam scores outside the valid range
    target = config.TARGET_COLUMN

    valid_scores = cleaned[target].between(
        config.SCORE_MIN,
        config.SCORE_MAX,
    )

    removed_scores = int((~valid_scores).sum())

    if removed_scores:
        cleaned = cleaned[valid_scores]

        logger.info(
            "Removed %d rows with invalid %s",
            removed_scores,
            target,
        )

    if cleaned.empty:
        raise ValueError("No rows left after removing invalid exam scores")

    # 6. Remove rows with invalid numeric feature values.
    # Missing values are allowed here because they will be filled later.
    valid_rows = pd.Series(True, index=cleaned.index)

    for col, (low, high) in config.NUMERIC_RANGES.items():

        valid_values = (
            cleaned[col].isna()
            | cleaned[col].between(low, high)
        )

        invalid_count = int((~valid_values).sum())

        if invalid_count:
            valid_rows &= valid_values

            logger.info(
                "Marked %d rows with invalid values in '%s' for removal",
                invalid_count,
                col,
            )

    # 7. Remove rows with unexpected categorical values.
    # Missing values are allowed here because they will be filled later.
    for col, allowed in config.CATEGORICAL_VALUES.items():

        valid_values = (
            cleaned[col].isna()
            | cleaned[col].isin(allowed)
        )

        invalid_count = int((~valid_values).sum())

        if invalid_count:
            valid_rows &= valid_values

            logger.info(
                "Marked %d rows with unexpected values in '%s' for removal",
                invalid_count,
                col,
            )

    removed_invalid = int((~valid_rows).sum())

    if removed_invalid:
        cleaned = cleaned[valid_rows]

        logger.info(
            "Removed %d rows with invalid feature values",
            removed_invalid,
        )

    if cleaned.empty:
        raise ValueError("No rows left after removing invalid feature values")

    # 8. Fill missing numeric values with the median.
    # Invalid values have already been removed, so they cannot affect
    # the median calculation.
    for col in config.NUMERIC_RANGES:

        n_missing = int(cleaned[col].isna().sum())

        if n_missing == 0:
            continue

        median = cleaned[col].median()

        if pd.isna(median):
            raise ValueError(
                f"Cannot fill missing values in '{col}': "
                "the column has no valid numeric values"
            )

        cleaned[col] = cleaned[col].fillna(median)

        logger.info(
            "Filled %d missing values in '%s' with the median",
            n_missing,
            col,
        )

    # 9. Fill missing categorical values with the mode.
    # Unexpected categories have already been removed.
    for col in config.CATEGORICAL_VALUES:

        n_missing = int(cleaned[col].isna().sum())

        if n_missing == 0:
            continue

        modes = cleaned[col].mode()

        if modes.empty:
            raise ValueError(
                f"Cannot fill missing values in '{col}': "
                "the column has no valid categorical values"
            )

        mode_value = modes.iloc[0]

        cleaned[col] = cleaned[col].fillna(mode_value)

        logger.info(
            "Filled %d missing values in '%s' with the mode",
            n_missing,
            col,
        )

    # 10. Remove exact duplicate rows
    duplicates = int(cleaned.duplicated().sum())

    if duplicates:
        cleaned = cleaned.drop_duplicates()

        logger.info(
            "Removed %d duplicate rows",
            duplicates,
        )

    # 11. Reset the index
    cleaned = cleaned.reset_index(drop=True)

    logger.info(
        "Cleaning finished: %d -> %d rows",
        rows_before,
        len(cleaned),
    )

    return cleaned


if __name__ == "__main__":

    from Load import load_data
    from Validate import validate_data

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    raw = load_data()
    cleaned = clean_data(raw)

    print("Before:", raw.shape, "| After:", cleaned.shape)
    print("Missing values left:", int(cleaned.isna().sum().sum()))
    print("Max Exam_Score:", cleaned[config.TARGET_COLUMN].max())

    result = validate_data(cleaned)

    print("Validation after cleaning:", result["is_valid"])
    print("Remaining errors:", result["errors"])
    print("Remaining warnings:", result["warnings"])
