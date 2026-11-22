
import pandas as pd
from Clean import clean_data


def test_clean_data():
    # Create sample data for testing
    data = {
        "Teacher_Quality": ["High", None, "Medium"],
        "Parental_Education_Level": ["College", None, "High School"],
        "Distance_from_Home": ["Near", "Far", None],
        "Exam_Score": [85, 70, 101]
    }

    df = pd.DataFrame(data)

    # Run the cleaning function
    cleaned_df = clean_data(df)

    # Test 1: Check missing values
    assert cleaned_df.isnull().sum().sum() == 0

    # Test 2: Check invalid exam scores
    assert cleaned_df["Exam_Score"].between(0, 100).all()

    # Test 3: Check invalid row was removed
    assert len(cleaned_df) == 2

    # Test 4: Check original data was not changed
    assert len(df) == 3

    print("\nAll tests passed successfully!")


if __name__ == "__main__":
    test_clean_data()
