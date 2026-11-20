
"""
Visualize Module

Generates and saves visualizations for the cleaned
student performance dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import logging

from config import VISUALIZATIONS_DIR


logger = logging.getLogger(__name__)


def visualize_data(df):
    """
    Generate and save visualizations for student performance data.

    The function creates:
    1. Exam score distribution histogram.
    2. Study hours versus exam score scatter plot.
    3. Attendance versus exam score scatter plot.
    4. Average exam score by motivation level bar chart.

    Args:
        df (pd.DataFrame): Cleaned student performance dataset.

    Returns:
        pd.DataFrame: Original DataFrame without modifications.

    Raises:
        TypeError: If the input is not a pandas DataFrame.
        ValueError: If the dataset is empty or required columns are missing.
    """

    logger.info("Starting data visualization.")

    # Check input data
    if not isinstance(df, pd.DataFrame):
        logger.error("Invalid input: Expected a pandas DataFrame.")
        raise TypeError("Visualize requires a pandas DataFrame.")

    if df.empty:
        logger.error("Cannot visualize an empty dataset.")
        raise ValueError("Cannot visualize an empty dataset.")

    # Check required columns
    required_columns = [
        "Exam_Score",
        "Hours_Studied",
        "Attendance",
        "Motivation_Level"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        error_message = (
            f"Missing required columns for visualization: {missing_columns}"
        )
        logger.error(error_message)
        raise ValueError(error_message)

    logger.info("Input data validation completed successfully.")
    logger.info("Dataset shape: %s", df.shape)

    # Create folder for visualizations
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Visualization directory is ready: %s",
        VISUALIZATIONS_DIR
    )

    # 1. Histogram: Exam Score Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["Exam_Score"], bins=10, edgecolor="black")
    plt.title("Distribution of Exam Scores")
    plt.xlabel("Exam Score")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "exam_score_histogram.png")
    plt.show()
    plt.close()

    logger.info("Saved exam score histogram.")

    # 2. Scatter Plot: Hours Studied vs Exam Score
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Hours_Studied"], df["Exam_Score"])
    plt.title("Hours Studied vs Exam Score")
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "study_hours_scatter.png")
    plt.show()
    plt.close()

    logger.info("Saved study hours scatter plot.")

    # 3. Scatter Plot: Attendance vs Exam Score
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Attendance"], df["Exam_Score"])
    plt.title("Attendance vs Exam Score")
    plt.xlabel("Attendance")
    plt.ylabel("Exam Score")
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "attendance_scatter.png")
    plt.show()
    plt.close()

    logger.info("Saved attendance scatter plot.")

    # 4. Bar Chart: Average Exam Score by Motivation
    average_scores = df.groupby("Motivation_Level")["Exam_Score"].mean()

    plt.figure(figsize=(8, 5))
    average_scores.plot(kind="bar", edgecolor="black")
    plt.title("Average Exam Score by Motivation Level")
    plt.xlabel("Motivation Level")
    plt.ylabel("Average Exam Score")
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "motivation_bar_chart.png")
    plt.show()
    plt.close()

    logger.info("Saved motivation bar chart.")

    logger.info("All visualizations generated successfully.")

    return df


if __name__ == "__main__":
    from Load import load_data
    from Clean import clean_data

    try:
        df = load_data()
        df_clean = clean_data(df)
        visualize_data(df_clean)

    except Exception:
        logger.exception("An error occurred during visualization.")
        raise
