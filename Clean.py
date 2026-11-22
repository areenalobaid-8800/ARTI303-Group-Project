
"""
Clean Module

Handles missing values and invalid exam scores
in the student performance dataset.
"""

import logging
import pandas as pd
from Load import load_data


# Configure logging
logging.basicConfig(
    filename="project.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

logger = logging.getLogger(__name__)


# Required columns for cleaning
REQUIRED_COLUMNS = [
    "Teacher_Quality",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Exam_Score"
]


# Categorical columns that require missing-value handling
CATEGORICAL_COLUMNS = [
    "Teacher_Quality",
    "Parental_Education_Level",
    "Distance_from_Home"
]


def clean_data(df):
    """
    Clean the student performance dataset.

    The function:
    1. Checks the input DataFrame and required columns.
    2. Creates a copy to preserve the original dataset.
    3. Fills missing categorical values using the mode.
    4. Removes rows with invalid exam scores (outside 0-100).
    5. Reports the cleaning results.

    Args:
        df (pd.DataFrame): Raw dataset to clean.

    Returns:
        pd.DataFrame: Cleaned dataset.

    Raises:
        TypeError: If the input is not a pandas DataFrame
                   or Exam_Score is not numerical.
        ValueError: If the dataset is empty, required columns
                    are missing, or a mode cannot be calculated.
    """

    logger.info("Starting data cleaning.")

    # 1. Validate input
    if not isinstance(df, pd.DataFrame):
        logger.error("Invalid input: Expected a pandas DataFrame.")
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        logger.error("Cannot clean an empty dataset.")
        raise ValueError("The dataset is empty.")

    # 2. Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        error_message = f"Missing required columns: {missing_columns}"
        logger.error(error_message)
        raise ValueError(error_message)

    # Check target column data type
    if not pd.api.types.is_numeric_dtype(df["Exam_Score"]):
        logger.error("Exam_Score must contain numerical values.")
        raise TypeError("Exam_Score must be numerical.")

    # 3. Create a copy to preserve the original dataset
    df_clean = df.copy()

    print("\n========== STARTING DATA CLEANING ==========")
    print("Original dataset shape:", df_clean.shape)

    logger.info("Original dataset shape: %s", df_clean.shape)

    # 4. Handle missing categorical values using mode
    print("\n--- Handling Missing Values ---")

    for col in CATEGORICAL_COLUMNS:

        missing_before = df_clean[col].isnull().sum()

        if missing_before > 0:

            mode_values = df_clean[col].mode()

            if mode_values.empty:
                error_message = (
                    f"Cannot fill missing values in '{col}': "
                    "No valid values available to calculate the mode."
                )

                logger.error(error_message)
                raise ValueError(error_message)

            mode_value = mode_values[0]

            df_clean[col] = df_clean[col].fillna(mode_value)

            print(
                f"{col}: Filled {missing_before} missing values "
                f"with '{mode_value}'."
            )

            logger.info(
                "Filled %s missing values in %s using mode: %s.",
                missing_before,
                col,
                mode_value
            )

        else:
            logger.info("No missing values found in %s.", col)

    # 5. Remove invalid exam scores
    print("\n--- Removing Invalid Exam Scores ---")

    invalid_scores = ~df_clean["Exam_Score"].between(0, 100)

    invalid_count = invalid_scores.sum()

    df_clean = df_clean.loc[~invalid_scores].copy()

    print(f"Removed {invalid_count} rows with invalid exam scores.")

    logger.info(
        "Removed %s rows with invalid exam scores.",
        invalid_count
    )

    # 6. Check the cleaning results
    remaining_missing = df_clean.isnull().sum().sum()

    remaining_invalid = (
        ~df_clean["Exam_Score"].between(0, 100)
    ).sum()

    print("\n========== CLEANING SUMMARY ==========")
    print("Final dataset shape:", df_clean.shape)
    print("Remaining missing values:", remaining_missing)
    print("Remaining invalid exam scores:", remaining_invalid)

    print("\n========== DATA CLEANING COMPLETED ==========")

    logger.info("Final dataset shape: %s", df_clean.shape)
    logger.info("Remaining missing values: %s", remaining_missing)
    logger.info("Remaining invalid exam scores: %s", remaining_invalid)

    if remaining_missing > 0:
        logger.warning(
            "Some missing values remain after cleaning: %s",
            remaining_missing
        )

    if remaining_invalid > 0:
        logger.warning(
            "Some invalid exam scores remain: %s",
            remaining_invalid
        )

    if remaining_missing == 0 and remaining_invalid == 0:
        logger.info("Data cleaning completed successfully.")
    else:
        logger.warning("Data cleaning completed with remaining issues.")

    return df_clean


if __name__ == "__main__":

    try:
        df = load_data()
        cleaned_df = clean_data(df)

    except FileNotFoundError as error:
        print(f"File Error: {error}")
        logger.exception("File error occurred during cleaning.")

    except (TypeError, ValueError) as error:
        print(f"Data Cleaning Error: {error}")
        logger.exception("Data cleaning error occurred.")

    except Exception as error:
        print(f"Unexpected Error: {error}")
        logger.exception("Unexpected error occurred during cleaning.")
