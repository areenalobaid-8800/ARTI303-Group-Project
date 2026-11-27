
"""Step 4 - Explore: exploratory data analysis (EDA)."""
import logging

import pandas as pd

import config

logger = logging.getLogger(__name__)


def explore_data(df):
    """Compute summary statistics and return them in a dictionary.

    The DataFrame is not modified.

    Returned keys:
      shape, numeric_summary, categorical_counts,
      correlations, mean_score_by_category

    Raises:
        TypeError: if the input is not a pandas DataFrame.
        ValueError: if the DataFrame is empty.
        KeyError: if required columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("explore_data expects a pandas DataFrame")

    if df.empty:
        raise ValueError("Cannot explore an empty DataFrame")

    # 1. Define the required project columns.
    numeric_cols = list(config.NUMERIC_RANGES)
    categorical_cols = list(config.CATEGORICAL_VALUES)
    target = config.TARGET_COLUMN

    required_cols = numeric_cols + categorical_cols + [target]

    # 2. Check that all required columns exist.
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # 3. Check that numeric columns and the target contain numeric data.
    for col in numeric_cols + [target]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(
                f"Column '{col}' must be numeric, but its type is {df[col].dtype}"
            )

    # 4. Generate exploratory data analysis results.
    results = {
        "shape": df.shape,

        "numeric_summary": df[numeric_cols + [target]].describe().T,

        "categorical_counts": {
            col: df[col].value_counts()
            for col in categorical_cols
        },

        "correlations": df[numeric_cols + [target]].corr(),

        "mean_score_by_category": {
            col: df.groupby(col)[target].mean().sort_values()
            for col in categorical_cols
        },
    }

    logger.info(
        "Exploration finished: %d rows, %d columns",
        *df.shape,
    )

    return results


def top_correlations(correlations, n=5):
    """Return the n features most correlated (by absolute value) with the target.

    Raises:
        TypeError: if correlations is not a pandas DataFrame or n is invalid.
        KeyError: if the target column is missing.
    """
    if not isinstance(correlations, pd.DataFrame):
        raise TypeError("correlations must be a pandas DataFrame")

    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise TypeError("n must be a non-negative integer")

    target = config.TARGET_COLUMN

    if target not in correlations.columns:
        raise KeyError(f"Target column '{target}' is missing from correlations")

    if target not in correlations.index:
        raise KeyError(f"Target row '{target}' is missing from correlations")

    series = correlations[target].drop(target)

    return series.reindex(
        series.abs().sort_values(ascending=False).index
    ).head(n)


if __name__ == "__main__":
    from Clean import clean_data
    from Load import load_data

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    data = clean_data(load_data())
    info = explore_data(data)

    print("Shape:", info["shape"])

    print("\nNumeric summary:")
    print(
        info["numeric_summary"][["mean", "std", "min", "max"]].round(2)
    )

    print("\nTop correlations with", config.TARGET_COLUMN)
    print(top_correlations(info["correlations"]).round(3))

    print("\nAverage score by Motivation_Level:")
    print(info["mean_score_by_category"]["Motivation_Level"].round(2))

    print("\nScore quartiles (25%, 50%, 75%):")
    print(data[config.TARGET_COLUMN].quantile([0.25, 0.5, 0.75]).to_string())
