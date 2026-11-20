
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


# Required columns in the dataset
REQUIRED_COLUMNS = [
    "Hours_Studied", "Attendance", "Parental_Involvement",
    "Access_to_Resources", "Extracurricular_Activities",
    "Sleep_Hours", "Previous_Scores", "Motivation_Level",
    "Internet_Access", "Tutoring_Sessions", "Family_Income",
    "Teacher_Quality", "School_Type", "Peer_Influence",
    "Physical_Activity", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home",
    "Gender", "Exam_Score"
]

TARGET_COLUMN = "Exam_Score"


# Numerical columns
NUMERICAL_COLUMNS = [
    "Hours_Studied", "Attendance", "Sleep_Hours",
    "Previous_Scores", "Tutoring_Sessions",
    "Physical_Activity", "Exam_Score"
]


# Categorical columns
CATEGORICAL_COLUMNS = [
    "Parental_Involvement", "Access_to_Resources",
    "Extracurricular_Activities", "Motivation_Level",
    "Internet_Access", "Family_Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender"
]


# Allowed values for categorical columns
ALLOWED_CATEGORIES = {
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Extracurricular_Activities": ["Yes", "No"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Internet_Access": ["Yes", "No"],
    "Family_Income": ["Low", "Medium", "High"],
    "Teacher_Quality": ["Low", "Medium", "High"],
    "School_Type": ["Public", "Private"],
    "Peer_Influence": ["Positive", "Negative", "Neutral"],
    "Learning_Disabilities": ["Yes", "No"],
    "Parental_Education_Level": [
        "High School", "College", "Postgraduate"
    ],
    "Distance_from_Home": ["Near", "Moderate", "Far"],
    "Gender": ["Male", "Female"]
}


# Expected numerical ranges
NUMERICAL_RANGES = {
    "Hours_Studied": (0, 168),
    "Attendance": (0, 100),
    "Sleep_Hours": (0, 24),
    "Previous_Scores": (0, 100),
    "Tutoring_Sessions": (0, None),
    "Physical_Activity": (0, 168),
    "Exam_Score": (0, 100)
}


def check_structure(df):
    """
    Check the dataset structure, required columns, and duplicate column names.

    Args:
        df (pd.DataFrame): Dataset to validate.

    Returns:
        bool: True if the dataset structure is valid, otherwise False.
    """
    print("\n========== 1. STRUCTURE & SCHEMA VALIDATION ==========")

    print(f"Dataset Dimensions: {df.shape[0]} Rows, {df.shape[1]} Columns")

    if df.empty:
        logger.warning("Dataset is empty.")
        print("Warning: Dataset is empty!")
        return False

    missing_cols = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_cols:
        logger.error("Missing required columns: %s", missing_cols)
        print("Warning: Missing required columns:", missing_cols)
    else:
        logger.info("All required columns are available.")
        print("All required columns are available.")

    if TARGET_COLUMN in df.columns:
        print(f"Target column '{TARGET_COLUMN}' is present.")
    else:
        logger.error("Target column '%s' is missing.", TARGET_COLUMN)
        print(f"Warning: Target column '{TARGET_COLUMN}' is missing!")

    duplicated_cols = df.columns[df.columns.duplicated()].tolist()

    if duplicated_cols:
        logger.error("Duplicate column names found: %s", duplicated_cols)
        print("Warning: Duplicate column names found:", duplicated_cols)

    if missing_cols or duplicated_cols:
        return False

    logger.info("Dataset structure validation passed.")
    return True


def check_data_types(df):
    """
    Check whether numerical and categorical columns have the expected data types.

    Args:
        df (pd.DataFrame): Dataset to validate.

    Returns:
        bool: True if all checked data types are valid.
    """
    print("\n========== 2. DATA TYPES VALIDATION ==========")

    is_valid = True

    for col in NUMERICAL_COLUMNS:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning("Column %s should be numerical.", col)
                print(
                    f"Warning: {col} should be numerical "
                    f"(Current: {df[col].dtype})."
                )
                is_valid = False
            else:
                print(f"{col}: Numerical data type is valid.")

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            dtype = df[col].dtype

            if not (
                pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)
            ):
                logger.warning("Column %s should be categorical.", col)
                print(
                    f"Warning: {col} should be categorical/string "
                    f"(Current: {dtype})."
                )
                is_valid = False
            else:
                print(f"{col}: Categorical data type is valid.")

    if is_valid:
        logger.info("Data types validation passed.")
    else:
        logger.warning("Data types validation failed.")

    return is_valid


def check_missing_values(df):
    """
    Check for actual missing values and hidden missing-value placeholders.

    Args:
        df (pd.DataFrame): Dataset to validate.

    Returns:
        bool: True if no missing values or placeholders are found.
    """
    print("\n========== 3. MISSING VALUES VALIDATION ==========")

    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()

    print(f"Total Missing Values: {total_missing}")

    if total_missing > 0:
        logger.warning("Found %s missing values.", total_missing)
        print("\nMissing Values per Column:")
        print(missing_counts[missing_counts > 0])
    else:
        logger.info("No actual missing values found.")
        print("No missing values found.")

    hidden_placeholders = ["", "N/A", "NA", "NULL", "NONE", "UNKNOWN"]
    found_hidden = False

    for col in df.select_dtypes(include=["object", "string"]).columns:
        values = df[col].dropna().astype(str).str.strip().str.upper()
        hidden_count = values.isin(hidden_placeholders).sum()

        if hidden_count > 0:
            found_hidden = True
            logger.warning(
                "Column %s contains %s hidden missing values.",
                col, hidden_count
            )
            print(
                f"Column '{col}': "
                f"{hidden_count} hidden missing values found."
            )

    if not found_hidden:
        print("No hidden missing value placeholders found.")

    is_valid = total_missing == 0 and not found_hidden

    if is_valid:
        logger.info("Missing values validation passed.")

    return is_valid


def check_duplicates(df):
    """
    Check whether the dataset contains duplicate rows.

    Args:
        df (pd.DataFrame): Dataset to validate.

    Returns:
        bool: True if no duplicate rows are found.
    """
    print("\n========== 4. DUPLICATE VALIDATION ==========")

    duplicate_rows = df.duplicated().sum()

    print(f"Duplicate Rows Count: {duplicate_rows}")

    if duplicate_rows > 0:
        logger.warning("Found %s duplicate rows.", duplicate_rows)
        return False

    logger.info("No duplicate rows found.")
    return True


def check_numerical_values(df):
    """
    Check numerical ranges, integer constraints, and statistical outliers.

    Args:
        df (pd.DataFrame): Dataset to validate.

    Returns:
        bool: True if no numerical range or integer violations are found.
    """
    print("\n========== 5. NUMERICAL VALUES & OUTLIERS VALIDATION ==========")

    is_valid = True

    print("\nRange Validation:")

    for col, (minimum, maximum) in NUMERICAL_RANGES.items():
        if col not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        if maximum is None:
            invalid = df[df[col] < minimum]
        else:
            invalid = df[
                (df[col] < minimum) | (df[col] > maximum)
            ]

        if len(invalid) > 0:
            logger.warning(
                "Column %s has %s out-of-range values.",
                col, len(invalid)
            )
            print(
                f"Warning: {col} has "
                f"{len(invalid)} out-of-range values."
            )
            is_valid = False
        else:
            print(f"{col}: No out-of-range values found.")

    if "Tutoring_Sessions" in df.columns:
        if pd.api.types.is_numeric_dtype(df["Tutoring_Sessions"]):
            values = df["Tutoring_Sessions"].dropna()
            invalid_count = (values % 1 != 0).sum()

            if invalid_count > 0:
                logger.warning(
                    "Tutoring_Sessions has %s non-integer values.",
                    invalid_count
                )
                print(
                    f"Warning: Tutoring_Sessions has "
                    f"{invalid_count} non-integer values."
                )
                is_valid = False
            else:
                print("Tutoring_Sessions: All values are whole numbers.")

    print("\nOutlier Summary (IQR Method):")

    for col in NUMERICAL_COLUMNS:
        if col not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        values = df[col].dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = values[
            (values < lower_bound) | (values > upper_bound)
        ]

        if len(outliers) > 0:
            logger.info(
                "Column %s has %s statistical outliers.",
                col, len(outliers)
            )
            print(
                f"{col}: {len(outliers)} statistical outliers detected."
            )
        else:
            print(f"{col}: No statistical outliers detected.")

    if is_valid:
        logger.info("Numerical range validation passed.")
    else:
        logger.warning("Numerical range validation failed.")

    return is_valid


def check_categorical_values(df):
    """
    Check categorical columns against their allowed values.

    Args:
        df (pd.DataFrame): Dataset to validate.

    Returns:
        bool: True if all categorical values are allowed.
    """
    print("\n========== 6. CATEGORICAL VALUES VALIDATION ==========")

    is_valid = True

    for col, allowed_values in ALLOWED_CATEGORIES.items():
        if col not in df.columns:
            continue

        actual_values = df[col].dropna().astype(str)

        allowed_set = {
            value.strip().lower() for value in allowed_values
        }

        invalid_mask = (
            ~actual_values.str.strip().str.lower().isin(allowed_set)
        )

        invalid_values = actual_values[invalid_mask].unique()

        if len(invalid_values) > 0:
            logger.warning(
                "Column %s has unexpected categories: %s",
                col, invalid_values
            )
            print(
                f"Warning: Column '{col}' has unexpected categories: "
                f"{invalid_values}"
            )
            is_valid = False
        else:
            print(f"{col}: All categorical values are valid.")

    if is_valid:
        logger.info("Categorical values validation passed.")

    return is_valid


def validate_data(df):
    """
    Run all dataset validation checks.

    Args:
        df (pd.DataFrame): Dataset to validate.

    Returns:
        bool: True if all validation checks pass, otherwise False.
    """
    logger.info("Starting data validation.")
    print("\n========== STARTING DATA VALIDATION ==========")

    if not isinstance(df, pd.DataFrame):
        logger.error("Invalid input: Expected a pandas DataFrame.")
        print("Validation Failed: Input must be a pandas DataFrame.")
        return False

    if not check_structure(df):
        logger.error("Validation failed due to invalid dataset structure.")
        print("\nValidation Failed: Invalid dataset structure.")
        return False

    data_types_valid = check_data_types(df)
    missing_values_valid = check_missing_values(df)
    duplicates_valid = check_duplicates(df)
    numerical_values_valid = check_numerical_values(df)
    categorical_values_valid = check_categorical_values(df)

    is_valid = all([
        data_types_valid,
        missing_values_valid,
        duplicates_valid,
        numerical_values_valid,
        categorical_values_valid
    ])

    if is_valid:
        logger.info("Data validation completed successfully.")
        print("\n========== DATA VALIDATION COMPLETED ==========")
        print("Validation Passed: No critical data errors found.")
    else:
        logger.warning("Data validation failed. Check the warnings above.")
        print("\n========== DATA VALIDATION FAILED ==========")
        print("Validation Failed: Please check the warnings above.")

    return is_valid


if __name__ == "__main__":
    try:
        df = load_data()
        validate_data(df)

    except Exception:
        logger.exception("An error occurred while running validation.")
        raise
