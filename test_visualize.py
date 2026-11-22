
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import Visualize


# Create sample dataset
def create_valid_data():

    data = {
        "Exam_Score": [60, 70, 80, 90],
        "Hours_Studied": [5, 10, 15, 20],
        "Attendance": [60, 70, 80, 90],
        "Motivation_Level": ["Low", "Medium", "High", "Medium"]
    }

    return pd.DataFrame(data)


# 1. Test successful visualization
def test_visualize_data():

    df = create_valid_data()
    original_df = df.copy()

    with TemporaryDirectory() as temp_dir:

        temp_path = Path(temp_dir)

        with patch("Visualize.VISUALIZATIONS_DIR", temp_path):
            with patch("Visualize.plt.show"):
                result = Visualize.visualize_data(df)

        expected_files = [
            "exam_score_histogram.png",
            "study_hours_scatter.png",
            "attendance_scatter.png",
            "motivation_bar_chart.png"
        ]

        for file_name in expected_files:
            assert (temp_path / file_name).exists()

        pd.testing.assert_frame_equal(result, original_df)

    print("Visualization success test passed!")


# 2. Test invalid input type
def test_visualize_invalid_input():

    try:
        Visualize.visualize_data(None)

    except TypeError as error:
        assert str(error) == (
            "Visualize requires a pandas DataFrame."
        )

    else:
        raise AssertionError(
            "Expected TypeError for invalid input."
        )

    print("Visualization invalid input test passed!")


# 3. Test empty dataset
def test_visualize_empty_data():

    empty_df = pd.DataFrame()

    try:
        Visualize.visualize_data(empty_df)

    except ValueError as error:
        assert str(error) == (
            "Cannot visualize an empty dataset."
        )

    else:
        raise AssertionError(
            "Expected ValueError for empty dataset."
        )

    print("Visualization empty data test passed!")


# 4. Test missing required columns
def test_visualize_missing_columns():

    df = create_valid_data()
    invalid_df = df.drop(columns=["Attendance"])

    try:
        Visualize.visualize_data(invalid_df)

    except ValueError as error:
        assert "Attendance" in str(error)

    else:
        raise AssertionError(
            "Expected ValueError for missing required columns."
        )

    print("Visualization missing columns test passed!")


# Run all tests
if __name__ == "__main__":

    test_visualize_data()
    test_visualize_invalid_input()
    test_visualize_empty_data()
    test_visualize_missing_columns()

    plt.close("all")

    print("\nAll Visualize tests passed successfully!")
