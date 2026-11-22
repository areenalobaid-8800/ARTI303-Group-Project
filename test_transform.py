
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import Transform


# Create a valid sample dataset
def create_valid_data():

    data = {
        "Hours_Studied": [i + 1 for i in range(20)],
        "Attendance": [60 + i for i in range(20)],
        "Parental_Involvement": [
            ["Low", "Medium", "High"][i % 3] for i in range(20)
        ],
        "Access_to_Resources": [
            ["Low", "Medium", "High"][i % 3] for i in range(20)
        ],
        "Extracurricular_Activities": [
            ["Yes", "No"][i % 2] for i in range(20)
        ],
        "Sleep_Hours": [6 + (i % 4) for i in range(20)],
        "Previous_Scores": [50 + i for i in range(20)],
        "Motivation_Level": [
            ["Low", "Medium", "High"][i % 3] for i in range(20)
        ],
        "Internet_Access": [
            ["Yes", "No"][i % 2] for i in range(20)
        ],
        "Tutoring_Sessions": [i % 5 for i in range(20)],
        "Family_Income": [
            ["Low", "Medium", "High"][i % 3] for i in range(20)
        ],
        "Teacher_Quality": [
            ["Low", "Medium", "High"][i % 3] for i in range(20)
        ],
        "School_Type": [
            ["Public", "Private"][i % 2] for i in range(20)
        ],
        "Peer_Influence": [
            ["Positive", "Negative", "Neutral"][i % 3]
            for i in range(20)
        ],
        "Physical_Activity": [i % 5 for i in range(20)],
        "Learning_Disabilities": [
            ["Yes", "No"][i % 2] for i in range(20)
        ],
        "Parental_Education_Level": [
            ["High School", "College", "Postgraduate"][i % 3]
            for i in range(20)
        ],
        "Distance_from_Home": [
            ["Near", "Moderate", "Far"][i % 3]
            for i in range(20)
        ],
        "Gender": [
            ["Male", "Female"][i % 2] for i in range(20)
        ],
        "Exam_Score": [60 + i for i in range(20)]
    }

    return pd.DataFrame(data)


# 1. Test raw numerical values
def test_check_raw_numerical_values():

    df = create_valid_data()

    assert Transform.check_raw_numerical_values(df) == True

    # Test out-of-range numerical values
    invalid_range_df = df.copy()
    invalid_range_df.loc[0, "Attendance"] = 150

    assert Transform.check_raw_numerical_values(
        invalid_range_df
    ) == False

    # Test non-integer tutoring sessions
    invalid_integer_df = df.copy()
    invalid_integer_df["Tutoring_Sessions"] = (
        invalid_integer_df["Tutoring_Sessions"].astype(float)
    )
    invalid_integer_df.loc[0, "Tutoring_Sessions"] = 1.5

    assert Transform.check_raw_numerical_values(
        invalid_integer_df
    ) == False

    print("Raw numerical validation tests passed!")


# 2. Test data preparation
def test_prepare_data():

    df = create_valid_data()

    with patch("Transform.load_data", return_value=df.copy()):
        with patch("Transform.clean_data", return_value=df.copy()):
            result = Transform.prepare_data()

    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape
    assert result.isnull().sum().sum() == 0

    pd.testing.assert_frame_equal(result, df)

    print("Data preparation test passed!")


# 3. Test data splitting
def test_split_data():

    df = create_valid_data()

    X_train, X_test, y_train, y_test = Transform.split_data(df)

    assert len(X_train) == 16
    assert len(X_test) == 4

    assert len(y_train) == 16
    assert len(y_test) == 4

    assert Transform.TARGET_COLUMN not in X_train.columns
    assert Transform.TARGET_COLUMN not in X_test.columns

    assert X_train.shape[1] == 19
    assert X_test.shape[1] == 19

    print("Data splitting test passed!")


# 4. Test feature transformation
def test_transform_data():

    df = create_valid_data()

    X = df[
        Transform.numerical_columns + Transform.categorical_columns
    ]

    X_train = X.iloc[:15]
    X_test = X.iloc[15:]

    X_train_transformed, X_test_transformed, feature_names = (
        Transform.transform_data(X_train, X_test)
    )

    assert X_train_transformed.shape[0] == len(X_train)
    assert X_test_transformed.shape[0] == len(X_test)

    assert X_train_transformed.shape[1] == len(feature_names)
    assert X_test_transformed.shape[1] == len(feature_names)

    assert np.isfinite(X_train_transformed).all()
    assert np.isfinite(X_test_transformed).all()

    print("Feature transformation test passed!")


# 5. Test DataFrame creation
def test_create_dataframes():

    df = create_valid_data()

    X = df[
        Transform.numerical_columns + Transform.categorical_columns
    ]

    X_train = X.iloc[:15]
    X_test = X.iloc[15:]

    y_train = df[Transform.TARGET_COLUMN].iloc[:15]
    y_test = df[Transform.TARGET_COLUMN].iloc[15:]

    X_train_transformed, X_test_transformed, feature_names = (
        Transform.transform_data(X_train, X_test)
    )

    train_df, test_df = Transform.create_dataframes(
        X_train_transformed,
        X_test_transformed,
        feature_names,
        X_train,
        X_test,
        y_train,
        y_test
    )

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    assert len(train_df) == 15
    assert len(test_df) == 5

    assert Transform.TARGET_COLUMN in train_df.columns
    assert Transform.TARGET_COLUMN in test_df.columns

    assert train_df.index.equals(X_train.index)
    assert test_df.index.equals(X_test.index)

    pd.testing.assert_series_equal(
        train_df[Transform.TARGET_COLUMN],
        y_train,
        check_names=False
    )

    pd.testing.assert_series_equal(
        test_df[Transform.TARGET_COLUMN],
        y_test,
        check_names=False
    )

    print("DataFrame creation test passed!")


# 6. Test exporting transformed data
def test_export_data():

    train_df = pd.DataFrame({
        "Feature_1": [1.0, 2.0],
        "Exam_Score": [60, 70]
    })

    test_df = pd.DataFrame({
        "Feature_1": [3.0],
        "Exam_Score": [80]
    })

    with TemporaryDirectory() as temp_dir:

        temp_path = Path(temp_dir)

        with patch("Transform.PROCESSED_DATA_DIR", temp_path):
            Transform.export_data(train_df, test_df)

        train_path = temp_path / "train_transformed.csv"
        test_path = temp_path / "test_transformed.csv"

        assert train_path.exists()
        assert test_path.exists()

        saved_train = pd.read_csv(train_path)
        saved_test = pd.read_csv(test_path)

        pd.testing.assert_frame_equal(saved_train, train_df)
        pd.testing.assert_frame_equal(saved_test, test_df)

    print("Data export test passed!")


# 7. Test summary generation
def test_generate_summary():

    df = create_valid_data()

    train_df = df.iloc[:16].copy()
    test_df = df.iloc[16:].copy()

    feature_names = ["Feature_1", "Feature_2"]

    with TemporaryDirectory() as temp_dir:

        temp_path = Path(temp_dir)

        with patch("Transform.PROCESSED_DATA_DIR", temp_path):
            summary_df = Transform.generate_summary(
                df,
                train_df,
                test_df,
                feature_names
            )

        summary_path = temp_path / "summary.csv"

        assert summary_path.exists()
        assert isinstance(summary_df, pd.DataFrame)

        summary_values = dict(
            zip(summary_df["Metric"], summary_df["Value"])
        )

        assert summary_values["Training Rows"] == 16
        assert summary_values["Testing Rows"] == 4
        assert summary_values["Total Rows"] == 20
        assert summary_values["Number of Features"] == 2
        assert summary_values["Training Missing Values"] == 0
        assert summary_values["Testing Missing Values"] == 0

        assert summary_values["Average Exam Score"] == df[
            "Exam_Score"
        ].mean()

    print("Summary generation test passed!")


# Run all tests
if __name__ == "__main__":

    test_check_raw_numerical_values()
    test_prepare_data()
    test_split_data()
    test_transform_data()
    test_create_dataframes()
    test_export_data()
    test_generate_summary()

    print("\nAll Transform tests passed successfully!")
