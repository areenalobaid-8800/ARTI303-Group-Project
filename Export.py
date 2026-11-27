
"""Step 7 - Export: save transformed data and the fitted preprocessor."""

import logging
from pathlib import Path

import joblib
import pandas as pd

import config

logger = logging.getLogger(__name__)


def export_data(train_df, test_df, preprocessor):
    """Export transformed datasets and the fitted preprocessor.

    Returns a dictionary containing the paths of the exported files.
    """

    if not isinstance(train_df, pd.DataFrame):
        raise TypeError("train_df must be a pandas DataFrame")

    if not isinstance(test_df, pd.DataFrame):
        raise TypeError("test_df must be a pandas DataFrame")

    if train_df.empty or test_df.empty:
        raise ValueError("Cannot export empty datasets")

    if not hasattr(preprocessor, "feature_names_in_"):
        raise ValueError("Preprocessor must be fitted before export")

    required_target = config.TARGET_COLUMN

    if required_target not in train_df.columns:
        raise KeyError(f"Missing target column in training data: {required_target}")

    if required_target not in test_df.columns:
        raise KeyError(f"Missing target column in testing data: {required_target}")

    processed_dir = Path(config.PROCESSED_DATA_DIR)
    models_dir = Path(config.MODELS_DIR)

    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    preprocessor_path = models_dir / "preprocessor.joblib"

    try:
        processed_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        joblib.dump(preprocessor, preprocessor_path)

        logger.info(
            "Export completed: train=%d rows, test=%d rows",
            len(train_df),
            len(test_df),
        )

        logger.info("Saved training data: %s", train_path)
        logger.info("Saved testing data: %s", test_path)
        logger.info("Saved preprocessor: %s", preprocessor_path)

    except (OSError, ValueError, TypeError):
        logger.exception("Failed to export project data")
        raise

    return {
        "train_data": train_path,
        "test_data": test_path,
        "preprocessor": preprocessor_path,
    }


if __name__ == "__main__":
    from Clean import clean_data
    from Load import load_data
    from Transform import transform_data

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    data = clean_data(load_data())

    train, test, preprocessor = transform_data(data)

    exported_files = export_data(train, test, preprocessor)

    print("\nExport completed successfully:")

    for name, path in exported_files.items():
        print(f"{name}: {path}")
