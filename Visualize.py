
import pandas as pd
import matplotlib.pyplot as plt
import os


def visualize_data(df):

    # Check input data
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Visualize requires a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot visualize an empty dataset.")

    # Create folder for visualizations
    os.makedirs("visualizations", exist_ok=True)

    # 1. Histogram: Exam Score Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["Exam_Score"], bins=10, edgecolor="black")
    plt.title("Distribution of Exam Scores")
    plt.xlabel("Exam Score")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig("visualizations/exam_score_histogram.png")
    plt.show()

    # 2. Scatter Plot: Hours Studied vs Exam Score
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Hours_Studied"], df["Exam_Score"])
    plt.title("Hours Studied vs Exam Score")
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.tight_layout()
    plt.savefig("visualizations/study_hours_scatter.png")
    plt.show()

    # 3. Scatter Plot: Attendance vs Exam Score
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Attendance"], df["Exam_Score"])
    plt.title("Attendance vs Exam Score")
    plt.xlabel("Attendance")
    plt.ylabel("Exam Score")
    plt.tight_layout()
    plt.savefig("visualizations/attendance_scatter.png")
    plt.show()

    # 4. Bar Chart: Average Exam Score by Motivation
    average_scores = df.groupby("Motivation_Level")["Exam_Score"].mean()

    plt.figure(figsize=(8, 5))
    average_scores.plot(kind="bar", edgecolor="black")
    plt.title("Average Exam Score by Motivation Level")
    plt.xlabel("Motivation Level")
    plt.ylabel("Average Exam Score")
    plt.tight_layout()
    plt.savefig("visualizations/motivation_bar_chart.png")
    plt.show()

    return df


if __name__ == "__main__":
    from Load import load_data
    from Clean import clean_data

    df = load_data()
    df_clean = clean_data(df)
    visualize_data(df_clean)
