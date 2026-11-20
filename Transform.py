import pandas as pd
from Validate import validate_data
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from Load import load_data
from Clean import clean_data


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


TARGET_COLUMN = "Exam_Score"
OUTPUT_DIR = Path("data/processed")


def prepare_data():
    df = load_data()

    if not validate_data(df):
        raise ValueError("Data validation failed.")

    df = clean_data(df)

    return df


# 4. Split data
def split_data(df):

    X = df[numerical_columns + categorical_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# 5. Transform data
def transform_data(X_train, X_test):

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("cat", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ), categorical_columns)
        ]
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    return X_train_transformed, X_test_transformed, feature_names


# 6. Create final DataFrames
def create_dataframes(
    X_train_transformed,
    X_test_transformed,
    feature_names,
    X_train,
    X_test,
    y_train,
    y_test
):

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

    return train_df, test_df


# 7. Export transformed data
def export_data(train_df, test_df):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        OUTPUT_DIR / "train_transformed.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "test_transformed.csv",
        index=False
    )

    print("\nData transformation completed successfully!")
    print("Training data shape:", train_df.shape)
    print("Testing data shape:", test_df.shape)


# 8. Generate data summary
def generate_summary(df, train_df, test_df, feature_names):

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

    summary_df.to_csv(
        OUTPUT_DIR / "summary.csv",
        index=False
    )

    print("\nData Summary:")
    print(summary_df)

    return summary_df


# 9. Main function
def main():

    df = prepare_data()

    X_train, X_test, y_train, y_test = split_data(df)

    X_train_transformed, X_test_transformed, feature_names = transform_data(
        X_train,
        X_test
    )

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


if __name__ == "__main__":
    main()