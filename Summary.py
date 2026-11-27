
"""Step 8 - Summary: generate and save a summary of the cleaned dataset."""

import json
import logging
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)


def _get_performance_level(score):
    """Convert an exam score into its performance level."""

    if score <= 60:
        return "Low"

    if score < 80:
        return "Medium"

    return "High"


def generate_summary(df):
    """Generate a summary containing dataset statistics and distributions."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")

    if df.empty:
        raise ValueError("Cannot summarize an empty DataFrame")

    required_cols = (
        list(config.NUMERIC_RANGES)
        + list(config.CATEGORICAL_VALUES)
        + [config.TARGET_COLUMN]
    )

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    target = config.TARGET_COLUMN

    for col in list(config.NUMERIC_RANGES) + [target]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric")

    # Summarize numeric features.
    numeric_summary = {}

    for col in list(config.NUMERIC_RANGES) + [target]:
        numeric_summary[col] = {
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }

    # Summarize categorical features.
    categorical_summary = {}

    for col in config.CATEGORICAL_VALUES:
        counts = df[col].value_counts(dropna=False)

        categorical_summary[col] = {
            str(category): int(count)
            for category, count in counts.items()
        }

    # Count performance levels.
    # Missing target values are excluded from the distribution.
    performance_levels = df[target].dropna().apply(
        _get_performance_level
    )

    performance_distribution = {
        level: int((performance_levels == level).sum())
        for level in ["Low", "Medium", "High"]
    }

    summary = {
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "missing_values": {
                col: int(count)
                for col, count in df.isna().sum().items()
            },
            "duplicate_rows": int(df.duplicated().sum()),
        },
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "performance_level_distribution": performance_distribution,
    }

    logger.info(
        "Summary generated successfully for %d rows and %d columns",
        df.shape[0],
        df.shape[1],
    )

    return summary


def save_summary(summary):
    """Save the generated summary as a JSON file."""

    if not isinstance(summary, dict) or not summary:
        raise ValueError("summary must be a non-empty dictionary")

    output_dir = Path(config.PROCESSED_DATA_DIR)
    output_path = output_dir / "data_summary.json"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                summary,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False,
            )

        logger.info("Saved data summary: %s", output_path)

    except (OSError, TypeError, ValueError):
        logger.exception("Failed to save data summary")
        raise

    return output_path


if __name__ == "__main__":
    from Clean import clean_data
    from Load import load_data

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    data = clean_data(load_data())

    summary = generate_summary(data)
    summary_path = save_summary(summary)

    print("\nData Summary")
    print("------------")
    print("Rows:", summary["dataset"]["rows"])
    print("Columns:", summary["dataset"]["columns"])
    print("Duplicate rows:", summary["dataset"]["duplicate_rows"])

    print("\nExam Score Statistics:")
    for name, value in summary["numeric_summary"][config.TARGET_COLUMN].items():
        print(f"{name}: {value:.2f}")

    print("\nPerformance Level Distribution:")
    for level, count in summary["performance_level_distribution"].items():
        print(f"{level}: {count}")

    print("\nSummary saved to:")
    print(summary_path)
