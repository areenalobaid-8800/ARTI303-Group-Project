
"""Step 2 - Validate: check structure, types, missing values, duplicates and ranges."""

import logging

import pandas as pd

import config

logger = logging.getLogger(__name__)


def validate_data(df):
    """Check the DataFrame and return a report dictionary.

    The DataFrame is never modified. Problems are split into:
      - errors: serious problems (missing columns, wrong data types)
      - warnings: problems that the cleaning step can fix (missing values,
                  duplicates, invalid numeric ranges, unexpected categories,
                  and exam scores outside the valid range)
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("validate_data expects a pandas DataFrame")

    if df.empty:
        raise ValueError("Cannot validate an empty DataFrame")

    errors, warnings = [], []

    required = (
        list(config.NUMERIC_RANGES)
        + list(config.CATEGORICAL_VALUES)
        + [config.TARGET_COLUMN]
    )

    # 1. Required columns
    missing_cols = [c for c in required if c not in df.columns]

    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # 2. Data types: numeric columns must contain numbers
    for col in list(config.NUMERIC_RANGES) + [config.TARGET_COLUMN]:

        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(
                f"Column '{col}' must be numeric but is {df[col].dtype}"
            )

    # 3. Missing values
    missing = df.isna().sum()

    missing = {
        col: int(n)
        for col, n in missing.items()
        if n > 0
    }

    if missing:
        warnings.append(f"Missing values: {missing}")

    # 4. Duplicate rows
    duplicates = int(df.duplicated().sum())

    if duplicates:
        warnings.append(f"{duplicates} duplicate rows")

    # 5. Numeric ranges
    for col, (low, high) in config.NUMERIC_RANGES.items():

        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):

            bad = int(
                ((df[col] < low) | (df[col] > high)).sum()
            )

            if bad:
                warnings.append(
                    f"Column '{col}': {bad} values outside [{low}, {high}]"
                )

    # 6. Exam score range
    target = config.TARGET_COLUMN

    if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):

        bad = int(
            (
                (df[target] < config.SCORE_MIN)
                | (df[target] > config.SCORE_MAX)
            ).sum()
        )

        if bad:
            warnings.append(
                f"{bad} rows with {target} outside "
                f"[{config.SCORE_MIN}, {config.SCORE_MAX}]"
            )

    # 7. Categorical values
    for col, allowed in config.CATEGORICAL_VALUES.items():

        if col in df.columns:

            unknown = sorted(
                set(df[col].dropna().astype(str)) - set(allowed)
            )

            if unknown:
                warnings.append(
                    f"Column '{col}': unexpected values {unknown}"
                )

    # 8. Validation report
    report = {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "missing_values": missing,
        "duplicate_rows": duplicates,
    }

    logger.info(
        "Validation finished: %d errors, %d warnings",
        len(errors),
        len(warnings),
    )

    return report


if __name__ == "__main__":

    from Load import load_data

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    result = validate_data(load_data())

    print("Valid:", result["is_valid"])

    for e in result["errors"]:
        print("ERROR:", e)

    for w in result["warnings"]:
        print("WARNING:", w)
