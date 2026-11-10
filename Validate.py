
import pandas as pd
from Load import load_data


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

#
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


# 1. Structure and Schema Validation
def check_structure(df):
    print("\n========== 1. STRUCTURE & SCHEMA VALIDATION ==========")

    print(f"Dataset Dimensions: {df.shape[0]} Rows, {df.shape[1]} Columns")

# check if the dataset is empty or not 
    if df.empty:
        print("Warning: Dataset is empty!")
        return False

    # Check required columns
    missing_cols = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_cols:
        print("Warning: Missing required columns:", missing_cols)
    else:
        print("All required columns are available.")

    # Check target column
    if TARGET_COLUMN in df.columns:
        print(f"Target column '{TARGET_COLUMN}' is present.")
    else:
        print(f"Warning: Target column '{TARGET_COLUMN}' is missing!")

    # Check duplicate column names
    duplicated_cols = df.columns[df.columns.duplicated()].tolist()

    if duplicated_cols:
        print("Warning: Duplicate column names found:", duplicated_cols)

    # Stop if required columns are missing or duplicated
    if missing_cols or duplicated_cols:
        return False

    return True


# 2. Data Types Validation
def check_data_types(df):
    print("\n========== 2. DATA TYPES VALIDATION ==========")

    # Check numerical columns
    for col in NUMERICAL_COLUMNS:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(
                    f"Warning: {col} should be numerical "
                    f"(Current: {df[col].dtype})."
                )
            else:
                print(f"{col}: Numerical data type is valid.")

    # Check categorical columns
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            dtype = df[col].dtype

            if not (
                pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)
            ):
                print(
                    f"Warning: {col} should be categorical/string "
                    f"(Current: {dtype})."
                )
            else:
                print(f"{col}: Categorical data type is valid.")


# 3. Missing Values Validation
def check_missing_values(df):
    print("\n========== 3. MISSING VALUES VALIDATION ==========")

    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()

    print(f"Total Missing Values: {total_missing}")

    if total_missing > 0:
        print("\nMissing Values per Column:")
        print(missing_counts[missing_counts > 0])
    else:
        print("No missing values found.")

    # Check hidden missing value placeholders
    hidden_placeholders = ["", "N/A", "NA", "NULL", "NONE", "UNKNOWN"]
    found_hidden = False

    for col in df.select_dtypes(include=["object", "string"]).columns:

        values = df[col].dropna().astype(str).str.strip().str.upper()

        hidden_count = values.isin(hidden_placeholders).sum()

        if hidden_count > 0:
            found_hidden = True
            print(
                f"Column '{col}': "
                f"{hidden_count} hidden missing values found."
            )

    if not found_hidden:
        print("No hidden missing value placeholders found.")


# 4. Duplicate Validation
def check_duplicates(df):
    print("\n========== 4. DUPLICATE VALIDATION ==========")

    duplicate_rows = df.duplicated().sum()

    print(f"Duplicate Rows Count: {duplicate_rows}")


# 5. Numerical Values and Outliers Validation
def check_numerical_values(df):
    print("\n========== 5. NUMERICAL VALUES & OUTLIERS VALIDATION ==========")

    # Range Validation
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
            print(
                f"Warning: {col} has "
                f"{len(invalid)} out-of-range values."
            )
        else:
            print(f"{col}: No out-of-range values found.")

    # Check whole numbers in Tutoring_Sessions
    if "Tutoring_Sessions" in df.columns:

        if pd.api.types.is_numeric_dtype(df["Tutoring_Sessions"]):

            values = df["Tutoring_Sessions"].dropna()

            invalid_count = (values % 1 != 0).sum()

            if invalid_count > 0:
                print(
                    f"Warning: Tutoring_Sessions has "
                    f"{invalid_count} non-integer values."
                )
            else:
                print(
                    "Tutoring_Sessions: All values are whole numbers."
                )

    # Statistical Outlier Detection using IQR
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
            print(
                f"{col}: {len(outliers)} statistical outliers detected."
            )
        else:
            print(f"{col}: No statistical outliers detected.")


# 6. Categorical Values Validation
def check_categorical_values(df):
    print("\n========== 6. CATEGORICAL VALUES VALIDATION ==========")

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
            print(
                f"Warning: Column '{col}' has unexpected categories: "
                f"{invalid_values}"
            )
        else:
            print(f"{col}: All categorical values are valid.")


# Main Validation Function
def validate_data(df):
    print("\n========== STARTING DATA VALIDATION ==========")

    structure_valid = check_structure(df)

    if not structure_valid:
        print("\nValidation stopped due to structural problems.")
        return

    check_data_types(df)
    check_missing_values(df)
    check_duplicates(df)
    check_numerical_values(df)
    check_categorical_values(df)

    print("\n========== DATA VALIDATION COMPLETED ==========")


if __name__ == "__main__":
    df = load_data()
    validate_data(df)
