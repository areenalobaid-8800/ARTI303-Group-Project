# استدعاء المكتبات الي نحتاجها
import pandas as pd
from pathlib import Path


def load_data():
    # Get the project directory
    project_dir = Path(__file__).resolve().parent

    # Get the raw data folder
    raw_dir = project_dir / "data" / "raw"

    # Find the CSV file
    csv_files = list(raw_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV file found in data/raw folder")

    # Load the dataset
    df = pd.read_csv(csv_files[0])

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    return df


if __name__ == "__main__":
    df = load_data()
