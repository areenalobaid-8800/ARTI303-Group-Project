
"""Step 1 - Load: read the dataset into a pandas DataFrame."""

import logging

import pandas as pd

import config

logger = logging.getLogger(__name__)


def load_data(path=config.RAW_DATA_FILE):
    """Load the CSV file and return it as a DataFrame.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is empty, malformed, or has an
                    unsupported text encoding.
    """
    try:
        df = pd.read_csv(path)

    except FileNotFoundError:
        logger.exception("Dataset not found: %s", path)
        raise

    except pd.errors.EmptyDataError as exc:
        logger.error("Dataset file is empty: %s", path)
        raise ValueError(f"Dataset file is empty: {path}") from exc

    except pd.errors.ParserError as exc:
        logger.error("Dataset could not be parsed: %s", path)
        raise ValueError(f"File is not a valid CSV: {path}") from exc

    except UnicodeDecodeError as exc:
        logger.error("Dataset encoding could not be decoded: %s", path)
        raise ValueError(f"Unsupported file encoding: {path}") from exc

    if df.empty:
        logger.error("Dataset has no rows: %s", path)
        raise ValueError("Dataset has no rows")

    logger.info(
        "Loaded %d rows and %d columns",
        df.shape[0],
        df.shape[1],
    )

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    data = load_data()

    print(data.shape)
    print(data.head())
