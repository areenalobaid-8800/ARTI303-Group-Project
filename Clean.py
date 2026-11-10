
import pandas as pd
import logging
from Load import load_data


# Configure logging
logging.basicConfig(
    filename="cleaning.log",
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


def clean_data(df):
    """
    Clean the dataset by handling missing values
    and removing invalid exam scores.
    """

    logger.info("Starting data cleaning.")

    # Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        error_message = f"Missing required columns: {missing_columns}"
        logger.error(error_message)
        raise ValueError(error_message)

    # Create a copy to preserve the original dataset
    df_clean = df.copy()

    print("\n========== STARTING DATA CLEANING ==========")
    print("Original dataset shape:", df_clean.shape)

    logger.info("Original dataset shape: %s", df_clean.shape)

    # 1. Handle missing categorical values using mode
    categorical_columns = [
        "Teacher_Quality",
        "Parental_Education_Level",
        "Distance_from_Home"
    ]

    print("\n--- Handling Missing Values ---")

    for col in categorical_columns:
        missing_before = df_clean[col].isnull().sum()

        if missing_before > 0:
            mode_value = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_value)

            print(
                f"{col}: Filled {missing_before} missing values "
                f"with '{mode_value}'."
            )

            logger.info(
                "Filled %s missing values in %s using %s.",
                missing_before,
                col,
                mode_value
            )

    # 2. Remove invalid exam scores
    print("\n--- Removing Invalid Exam Scores ---")

    invalid_scores = ~df_clean["Exam_Score"].between(0, 100)
    invalid_count = invalid_scores.sum()

    df_clean = df_clean.loc[~invalid_scores].copy()

    print(f"Removed {invalid_count} rows with invalid exam scores.")

    logger.info(
        "Removed %s rows with invalid exam scores.",
        invalid_count
    )

    # 3. Display cleaning results
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
    logger.info("Data cleaning completed successfully.")

    return df_clean


if __name__ == "__main__":
    try:
        df = load_data()
        cleaned_df = clean_data(df)

    except FileNotFoundError as error:
        print(f"File Error: {error}")
        logger.exception("File error occurred.")

    except ValueError as error:
        print(f"Data Validation Error: {error}")
        logger.exception("Data validation error occurred.")

    except Exception as error:
        print(f"Unexpected Error: {error}")
        logger.exception("Unexpected error occurred.")
