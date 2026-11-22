
import logging
import pandas as pd
from config import RAW_DATA_DIR


# Configure logging
logging.basicConfig(
    filename="project.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

logger = logging.getLogger(__name__)


def load_data():
    """
    Load the raw student performance dataset from the data/raw folder.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If no CSV file is found.
        Exception: If an error occurs while loading the dataset.
    """
    try:
        # Get the raw data folder
        raw_dir = RAW_DATA_DIR

        logger.info("Starting dataset loading process.")

        # Find the CSV file
        csv_files = list(raw_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                "No CSV file found in data/raw folder."
            )

        # Load the dataset
        df = pd.read_csv(csv_files[0])

        logger.info("Dataset loaded successfully.")
        logger.info("Dataset shape: %s", df.shape)

        print("Dataset loaded successfully!")
        print("Dataset shape:", df.shape)
        print("\nFirst 5 rows:")
        print(df.head())

        return df

    except FileNotFoundError as e:
        logger.error("Dataset file not found: %s", e)
        raise

    except Exception as e:
        logger.exception("An unexpected error occurred while loading data: %s", e)
        raise


if __name__ == "__main__":
    df = load_data()
