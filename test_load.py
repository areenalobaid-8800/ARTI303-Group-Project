
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import Load


def test_load_data_success():
    # Create a temporary folder and CSV file
    with TemporaryDirectory() as temp_dir:

        raw_dir = Path(temp_dir)

        sample_df = pd.DataFrame({
            "Hours_Studied": [10, 20, 30],
            "Attendance": [70, 80, 90],
            "Exam_Score": [60, 70, 80]
        })

        csv_path = raw_dir / "sample.csv"
        sample_df.to_csv(csv_path, index=False)

        # Replace the raw data directory with the temporary folder
        with patch("Load.RAW_DATA_DIR", raw_dir):
            loaded_df = Load.load_data()

        # Check that the loaded data is a DataFrame
        assert isinstance(loaded_df, pd.DataFrame)

        # Check that the loaded data matches the original data
        pd.testing.assert_frame_equal(loaded_df, sample_df)

        print("Load data success test passed!")


def test_load_data_no_csv():
    # Create an empty temporary folder
    with TemporaryDirectory() as temp_dir:

        raw_dir = Path(temp_dir)

        # Replace the raw data directory with the empty folder
        with patch("Load.RAW_DATA_DIR", raw_dir):

            try:
                Load.load_data()

            except FileNotFoundError as error:
                assert str(error) == (
                    "No CSV file found in data/raw folder."
                )

            else:
                raise AssertionError(
                    "Expected FileNotFoundError when no CSV file exists."
                )

        print("Load missing file test passed!")


def test_load_data_read_error():
    # Create a temporary folder with a CSV file
    with TemporaryDirectory() as temp_dir:

        raw_dir = Path(temp_dir)
        csv_path = raw_dir / "sample.csv"
        csv_path.write_text("sample data")

        # Simulate an error while reading the CSV file
        with patch("Load.RAW_DATA_DIR", raw_dir):
            with patch(
                "Load.pd.read_csv",
                side_effect=ValueError("Invalid CSV data")
            ):

                try:
                    Load.load_data()

                except ValueError as error:
                    assert str(error) == "Invalid CSV data"

                else:
                    raise AssertionError(
                        "Expected ValueError when CSV reading fails."
                    )

        print("Load read error test passed!")


if __name__ == "__main__":
    test_load_data_success()
    test_load_data_no_csv()
    test_load_data_read_error()

    print("\nAll Load tests passed successfully!")
