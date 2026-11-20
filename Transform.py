
"""
Transform Module

Prepares, splits, transforms, and exports the student
performance dataset for machine learning.
"""

import logging
import pandas as pd

from Validate import validate_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from Load import load_data
from Clean import clean_data

from config import (
    TARGET_COLUMN,
    PROCESSED_DATA_DIR,
    TEST_SIZE,
    RANDOM_STATE
)


logger = logging.getLogger(__name__)


# 1. Define numerical columns
numerical_columns = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity"
]


# 2. Define categorical columns
categorical_columns = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Motivation_Level",
    "Internet_Access",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender"
]


def prepare_data():
    """
    Load, validate, and clean the raw student dataset.

    Raw data validation warnings are logged because missing
    values and invalid scores are handled during cleaning.
    The cleaned dataset must pass validation.

    Returns:
        pd.DataFrame: Cleaned and validated dataset.

    Raises:
        ValueError: If the cleaned dataset fails validation.
    """

    logger.info("Starting data preparation.")

    # 1. Load the raw dataset
    df = load_data()

    # 2. Validate the raw dataset
    print("\nValidating raw dataset...")
    logger.info("Validating raw dataset.")

    if not validate_data(df):
        logger.warning(
            "Raw dataset contains validation issues. "
            "Proceeding to cleaning."
        )

    # 3. Clean the dataset
    df = clean_data(df)

    # 4. Validate the cleaned dataset
    print("\nValidating cleaned dataset...")
    logger.info("Validating cleaned dataset.")

    if not validate_data(df):
        logger.error("Data validation failed after cleaning.")
        raise ValueError("Data validation failed after cleaning.")

    print("\nData preparation completed successfully!")

    logger.info("Data preparation completed successfully.")
    logger.info("Cleaned dataset shape: %s", df.shape)

    return df


def split_data(df):
    """
    Split the dataset into training and testing sets.

    The target column is separated from the input features.

    Args:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        tuple: X_train, X_test, y_train, y_test.
    """

    logger.info("Starting train-test split.")

    X = df[numerical_columns + categorical_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    logger.info("Training rows: %s", len(X_train))
    logger.info("Testing rows: %s", len(X_test))

    print("\nData splitting completed successfully!")
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    return X_train, X_test, y_train, y_test


def transform_data(X_train, X_test):
    """
    Scale numerical features and encode categorical features.

    The preprocessor is fitted only on training data to prevent
    data leakage. The same transformations are applied to test data.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.

    Returns:
        tuple: Transformed training data, transformed testing data,
               and feature names.
    """

    logger.info("Starting feature transformation.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_columns
            )
        ]
    )

    # Fit only on training data to prevent data leakage
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Apply the fitted transformations to test data
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    logger.info("Feature transformation completed successfully.")
    logger.info("Number of transformed features: %s", len(feature_names))

    return X_train_transformed, X_test_transformed, feature_names


def create_dataframes(
    X_train_transformed,
    X_test_transformed,
    feature_names,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Create DataFrames from transformed features and add target values.

    Original row indices are preserved when creating the DataFrames.

    Returns:
        tuple: Transformed training and testing DataFrames.
    """

    logger.info("Creating final transformed DataFrames.")

    train_df = pd.DataFrame(
        X_train_transformed,
        columns=feature_names,
        index=X_train.index
    )

    test_df = pd.DataFrame(
        X_test_transformed,
        columns=feature_names,
        index=X_test.index
    )

    train_df[TARGET_COLUMN] = y_train
    test_df[TARGET_COLUMN] = y_test

    logger.info("Training DataFrame shape: %s", train_df.shape)
    logger.info("Testing DataFrame shape: %s", test_df.shape)

    return train_df, test_df


def export_data(train_df, test_df):
    """
    Export transformed training and testing datasets as CSV files.

    Files are saved in the configured processed data directory.
    """

    logger.info("Starting transformed data export.")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DATA_DIR / "train_transformed.csv"
    test_path = PROCESSED_DATA_DIR / "test_transformed.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nData transformation completed successfully!")
    print("Training data shape:", train_df.shape)
    print("Testing data shape:", test_df.shape)

    logger.info("Training data exported to: %s", train_path)
    logger.info("Testing data exported to: %s", test_path)
    logger.info("Transformed data export completed successfully.")


def generate_summary(df, train_df, test_df, feature_names):
    """
    Generate and export a summary of the processed dataset.

    The summary includes dataset sizes, feature count,
    missing values, and exam score statistics.

    Returns:
        pd.DataFrame: Summary of the processed dataset.
    """

    logger.info("Generating processed data summary.")

    summary = {
        "Training Rows": len(train_df),
        "Testing Rows": len(test_df),
        "Total Rows": len(train_df) + len(test_df),
        "Number of Features": len(feature_names),
        "Training Missing Values": train_df.isnull().sum().sum(),
        "Testing Missing Values": test_df.isnull().sum().sum(),
        "Average Exam Score": df[TARGET_COLUMN].mean(),
        "Minimum Exam Score": df[TARGET_COLUMN].min(),
        "Maximum Exam Score": df[TARGET_COLUMN].max()
    }

    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["Metric", "Value"]
    )

    summary_path = PROCESSED_DATA_DIR / "summary.csv"

    summary_df.to_csv(summary_path, index=False)

    print("\nData Summary:")
    print(summary_df)

    logger.info("Summary exported to: %s", summary_path)
    logger.info("Processed data summary generated successfully.")

    return summary_df


def main():
    """
    Execute the complete data preparation and transformation pipeline.

    Steps:
    1. Prepare and validate the dataset.
    2. Split data into training and testing sets.
    3. Transform numerical and categorical features.
    4. Create final DataFrames.
    5. Export transformed datasets.
    6. Generate the data summary.
    """

    logger.info("Starting the complete transformation pipeline.")

    try:
        df = prepare_data()

        X_train, X_test, y_train, y_test = split_data(df)

        (
            X_train_transformed,
            X_test_transformed,
            feature_names
        ) = transform_data(X_train, X_test)

        train_df, test_df = create_dataframes(
            X_train_transformed,
            X_test_transformed,
            feature_names,
            X_train,
            X_test,
            y_train,
            y_test
        )

        export_data(train_df, test_df)

        generate_summary(
            df,
            train_df,
            test_df,
            feature_names
        )

        logger.info("Transformation pipeline completed successfully.")

    except Exception:
        logger.exception("An error occurred in the transformation pipeline.")
        raise


if __name__ == "__main__":
    main()
