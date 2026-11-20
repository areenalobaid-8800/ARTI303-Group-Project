
import pandas as pd
from pandas.testing import assert_frame_equal
from Explore import explore_data


def test_explore_data():
    # Create a small sample dataset
    df = pd.DataFrame({
        "Hours_Studied": [10, 20, 30],
        "Attendance": [70, 80, 90],
        "Exam_Score": [60, 70, 80],
        "Gender": ["Male", "Female", "Male"],
        "Motivation_Level": ["Low", "Medium", "High"]
    })

    # Save original data for comparison
    original_df = df.copy(deep=True)

    # Run exploratory analysis
    result = explore_data(df)

    # Check that Explore returns the same data
    assert_frame_equal(result, original_df)

    # Check that Explore does not modify the original data
    assert_frame_equal(df, original_df)

    print("\nExplore test passed successfully!")


def test_explore_errors():
    # Test invalid input type
    try:
        explore_data(None)
    except TypeError as error:
        assert str(error) == "Explore requires a pandas DataFrame."
    else:
        raise AssertionError("Expected TypeError for invalid input.")

    # Test empty dataset
    try:
        explore_data(pd.DataFrame())
    except ValueError as error:
        assert str(error) == "Cannot explore an empty dataset."
    else:
        raise AssertionError("Expected ValueError for empty dataset.")

    # Test missing Exam_Score column
    try:
        explore_data(pd.DataFrame({"Hours_Studied": [10, 20]}))
    except ValueError as error:
        assert str(error) == "Exam_Score column is required for EDA."
    else:
        raise AssertionError("Expected ValueError for missing Exam_Score.")

    print("Explore error handling tests passed successfully!")


if __name__ == "__main__":
    test_explore_data()
    test_explore_errors()