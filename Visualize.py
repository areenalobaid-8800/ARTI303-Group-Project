
"""Step 5 - Visualize: create and save the project charts."""

import logging

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config

logger = logging.getLogger(__name__)


def _check(df, columns):
    """Validate the DataFrame and the columns required for a plot."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")

    if df.empty:
        raise ValueError("Cannot plot an empty DataFrame")

    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise KeyError(f"Missing columns for plot: {missing}")

    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric")


def plot_score_distribution(df):
    """Create a histogram showing the distribution of exam scores."""

    target = config.TARGET_COLUMN
    _check(df, [target])

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=target,
        bins=25,
        kde=True,
        color="#457b9d",
        ax=ax,
    )

    ax.set_title("Exam Score Distribution", fontsize=14)
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Number of Students")

    fig.tight_layout()

    return fig


def plot_scatter(df, feature):
    """Create a scatter plot showing a feature's relationship with exam scores."""

    target = config.TARGET_COLUMN
    _check(df, [feature, target])

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.regplot(
        data=df,
        x=feature,
        y=target,
        scatter_kws={"alpha": 0.25, "s": 12},
        line_kws={"color": "red"},
        ax=ax,
    )

    ax.set_title(
        f"{feature.replace('_', ' ')} vs Exam Score",
        fontsize=14,
    )
    ax.set_xlabel(feature.replace("_", " "))
    ax.set_ylabel("Exam Score")

    fig.tight_layout()

    return fig


def plot_mean_by_category(df, category):
    """Create a bar chart showing average exam scores by category."""

    target = config.TARGET_COLUMN

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")

    if df.empty:
        raise ValueError("Cannot plot an empty DataFrame")

    missing = [col for col in [category, target] if col not in df.columns]

    if missing:
        raise KeyError(f"Missing columns for plot: {missing}")

    if not pd.api.types.is_numeric_dtype(df[target]):
        raise TypeError(f"Column '{target}' must be numeric")

    if not (
        pd.api.types.is_object_dtype(df[category])
        or pd.api.types.is_string_dtype(df[category].dtype)
        or isinstance(df[category].dtype, pd.CategoricalDtype)
    ):
        raise TypeError(f"Column '{category}' must be categorical")

    means = df.groupby(category)[target].mean().sort_values()

    if means.empty:
        raise ValueError(f"No valid groups found for column '{category}'")

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        means.index.astype(str),
        means.values,
        color="#2a9d8f",
    )

    ax.set_ylim(0, max(100, means.max() + 5))

    for bar, value in zip(bars, means.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.5,
            f"{value:.1f}",
            ha="center",
            fontsize=10,
        )

    ax.set_title(
        f"Average Exam Score by {category.replace('_', ' ')}",
        fontsize=14,
    )
    ax.set_xlabel(category.replace("_", " "))
    ax.set_ylabel("Average Exam Score")

    fig.tight_layout()

    return fig


def plot_correlation_heatmap(df):
    """Create a heatmap showing correlations between numeric features."""

    numeric = list(config.NUMERIC_RANGES) + [config.TARGET_COLUMN]

    _check(df, numeric)

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        df[numeric].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title("Correlation Between Numeric Features", fontsize=14)

    fig.tight_layout()

    return fig


def save_figure(fig, filename):
    """Save a figure in the visualizations directory and close it."""

    try:
        if fig is None:
            raise ValueError("Cannot save an empty figure")

        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string")

        path = config.VISUALIZATIONS_DIR / filename

        config.VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved visualization: %s", path.name)

    except OSError:
        logger.exception("Failed to save visualization")
        raise

    finally:
        if fig is not None:
            plt.close(fig)

    return path


def generate_all_visualizations(df):
    """Generate and save all project visualizations."""

    plots = [
        (plot_score_distribution, "score_distribution.png", ()),
        (plot_scatter, "hours_vs_score.png", ("Hours_Studied",)),
        (plot_scatter, "attendance_vs_score.png", ("Attendance",)),
        (plot_mean_by_category, "motivation_vs_score.png", ("Motivation_Level",)),
        (plot_correlation_heatmap, "correlation_heatmap.png", ()),
    ]

    paths = []

    for plot_function, filename, args in plots:
        fig = None

        try:
            if args:
                fig = plot_function(df, *args)
            else:
                fig = plot_function(df)

            paths.append(save_figure(fig, filename))

        except Exception:
            if fig is not None:
                plt.close(fig)

            logger.exception("Failed to generate %s", filename)
            raise

    logger.info("Generated %d visualizations successfully", len(paths))

    return paths


if __name__ == "__main__":
    from Clean import clean_data
    from Load import load_data

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    data = clean_data(load_data())

    paths = generate_all_visualizations(data)

    print(f"\nCreated {len(paths)} charts in:")
    print(config.VISUALIZATIONS_DIR)

    for path in paths:
        print("-", path.name)
