import pandas as pd


def explore_data(df):
    """
    Perform Exploratory Data Analysis (EDA) on cleaned student data.

    The analysis includes:
    - Dataset dimensions
    - Numerical and categorical summary statistics
    - Correlations with Exam_Score
    - Categorical feature distributions
    - Average exam scores across categorical groups

    Parameters:
        df (pd.DataFrame): Cleaned student performance dataset.

    Returns:
        pd.DataFrame: Original DataFrame without modifications.
    """

    # Check that EDA receives valid input
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Explore requires a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot explore an empty dataset.")

    if "Exam_Score" not in df.columns:
        raise ValueError("Exam_Score column is required for EDA.")

    print("\n=== Exploratory Data Analysis (EDA) ===")

    # Display dataset dimensions
    print("\nDataset Shape:")
    print(df.shape)

    # Display numerical summary statistics
    print("\nNumerical Summary Statistics:")
    print(df.describe())

    # Display categorical summary statistics
    print("\nCategorical Summary Statistics:")
    print(df.describe(include=["object", "str"]))

    # Analyze correlation between numerical features
    print("\n=== Correlation with Exam Score ===")

    numeric_df = df.select_dtypes(include="number")

    correlations = numeric_df.corr()["Exam_Score"]
    correlations = correlations.drop("Exam_Score")
    correlations = correlations.sort_values(ascending=False)

    print(correlations)

    # Summarize the strongest numerical relationships
    print("\n=== Key EDA Findings ===")
    print(f"Average Exam Score: {df['Exam_Score'].mean():.2f}")

    strongest = correlations.abs().sort_values(ascending=False).head(3)

    print("\nTop 3 numerical features by absolute correlation:")
    for feature in strongest.index:
        print(f"{feature}: {correlations[feature]:.3f}")

    # Explore categorical feature distributions
    print("\n=== Categorical Feature Distributions ===")

    categorical_df = df.select_dtypes(include=["object", "str"])

    for column in categorical_df.columns:
        print(f"\n{column}:")
        print(categorical_df[column].value_counts())

    # Compare average exam scores across categorical groups
    print("\n=== Average Exam Score by Category ===")

    for column in categorical_df.columns:
        print(f"\n{column}:")
        average_scores = df.groupby(column)["Exam_Score"].mean()
        print(average_scores.sort_values(ascending=False).round(2))

    return df


if __name__ == "__main__":
    from Load import load_data
    from Clean import clean_data

    df = load_data()
    df_clean = clean_data(df)
    explore_data(df_clean)