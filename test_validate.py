
import pandas as pd
import Validate


# Create a valid sample dataset
def create_valid_data():

    data = {
        "Hours_Studied": [10, 20, 30, 40],
        "Attendance": [60, 70, 80, 90],
        "Parental_Involvement": ["Low", "Medium", "High", "Low"],
        "Access_to_Resources": ["Low", "Medium", "High", "Low"],
        "Extracurricular_Activities": ["Yes", "No", "Yes", "No"],
        "Sleep_Hours": [6, 7, 8, 9],
        "Previous_Scores": [60, 65, 70, 75],
        "Motivation_Level": ["Low", "Medium", "High", "Low"],
        "Internet_Access": ["Yes", "No", "Yes", "No"],
        "Tutoring_Sessions": [0, 1, 2, 3],
        "Family_Income": ["Low", "Medium", "High", "Low"],
        "Teacher_Quality": ["Low", "Medium", "High", "Low"],
        "School_Type": ["Public", "Private", "Public", "Private"],
        "Peer_Influence": ["Positive", "Negative", "Neutral", "Positive"],
        "Physical_Activity": [1, 2, 3, 4],
        "Learning_Disabilities": ["Yes", "No", "Yes", "No"],
        "Parental_Education_Level": [
            "High School", "College", "Postgraduate", "High School"
        ],
        "Distance_from_Home": ["Near", "Moderate", "Far", "Near"],
        "Gender": ["Male", "Female", "Male", "Female"],
        "Exam_Score": [60, 65, 70, 75]
    }

    return pd.DataFrame(data)


# 1. Test structure validation
def test_check_structure():

    df = create_valid_data()

    assert Validate.check_structure(df) == True

    invalid_df = df.drop(columns=["Exam_Score"])
    assert Validate.check_structure(invalid_df) == False

    empty_df = pd.DataFrame()
    assert Validate.check_structure(empty_df) == False

    print("Structure tests passed!")


# 2. Test data types validation
def test_check_data_types():

    df = create_valid_data()

    assert Validate.check_data_types(df) == True

    invalid_df = df.copy()
    invalid_df["Hours_Studied"] = invalid_df["Hours_Studied"].astype(str)

    assert Validate.check_data_types(invalid_df) == False

    print("Data types tests passed!")


# 3. Test missing values validation
def test_check_missing_values():

    df = create_valid_data()

    assert Validate.check_missing_values(df) == True

    missing_df = df.copy()
    missing_df.loc[0, "Teacher_Quality"] = None

    assert Validate.check_missing_values(missing_df) == False

    hidden_missing_df = df.copy()
    hidden_missing_df.loc[0, "Teacher_Quality"] = "N/A"

    assert Validate.check_missing_values(hidden_missing_df) == False

    print("Missing values tests passed!")


# 4. Test duplicate validation
def test_check_duplicates():

    df = create_valid_data()

    assert Validate.check_duplicates(df) == True

    duplicate_df = pd.concat(
        [df, df.iloc[[0]]],
        ignore_index=True
    )

    assert Validate.check_duplicates(duplicate_df) == False

    print("Duplicate tests passed!")


# 5. Test numerical values validation
def test_check_numerical_values():

    df = create_valid_data()

    assert Validate.check_numerical_values(df) == True

    # Test values outside the allowed range
    invalid_range_df = df.copy()
    invalid_range_df.loc[0, "Attendance"] = 150

    assert Validate.check_numerical_values(invalid_range_df) == False

    # Test non-integer tutoring sessions
    invalid_integer_df = df.copy()
    invalid_integer_df["Tutoring_Sessions"] = (
        invalid_integer_df["Tutoring_Sessions"].astype(float)
    )
    invalid_integer_df.loc[0, "Tutoring_Sessions"] = 1.5

    assert Validate.check_numerical_values(invalid_integer_df) == False

    print("Numerical values tests passed!")


# 6. Test categorical values validation
def test_check_categorical_values():

    df = create_valid_data()

    assert Validate.check_categorical_values(df) == True

    invalid_df = df.copy()
    invalid_df.loc[0, "Gender"] = "Unknown"

    assert Validate.check_categorical_values(invalid_df) == False

    print("Categorical values tests passed!")


# 7. Test complete validation
def test_validate_data():

    df = create_valid_data()

    assert Validate.validate_data(df) == True

    invalid_df = df.copy()
    invalid_df.loc[0, "Gender"] = "Unknown"

    assert Validate.validate_data(invalid_df) == False

    assert Validate.validate_data(None) == False

    print("Complete validation tests passed!")


# Run all tests
if __name__ == "__main__":

    test_check_structure()
    test_check_data_types()
    test_check_missing_values()
    test_check_duplicates()
    test_check_numerical_values()
    test_check_categorical_values()
    test_validate_data()

    print("\nAll Validate tests passed successfully!")
