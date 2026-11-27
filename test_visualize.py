
"""Tests for Step 5 - Visualize."""

import logging

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import config
import Visualize


@pytest.fixture
def valid_data():
    """Create a valid DataFrame using the project configuration."""

    data = {}

    for col, (low, high) in config.NUMERIC_RANGES.items():
        data[col] = [
            low + (high - low) * 0.25,
            low + (high - low) * 0.50,
            low + (high - low) * 0.75,
            low + (high - low) * 0.60,
        ]

    for col, allowed in config.CATEGORICAL_VALUES.items():
        values = [
            allowed[0],
            allowed[0],
            allowed[1],
            allowed[2] if len(allowed) > 2 else allowed[0],
        ]

        # Explicitly use object dtype for compatibility with Visualize.py.
        data[col] = pd.Series(values, dtype=object)

    data[config.TARGET_COLUMN] = [60, 70, 80, 90]

    return pd.DataFrame(data)


@pytest.fixture
def temporary_visualizations_dir(tmp_path, monkeypatch):
    """Use a temporary directory instead of the project visualization directory."""

    directory = tmp_path / "visualizations"
    monkeypatch.setattr(config, "VISUALIZATIONS_DIR", directory)

    return directory


# -------------------------
# Tests for _check()
# -------------------------

def test_check_valid_data(valid_data):
    """Test that valid data passes the plotting validation."""

    Visualize._check(valid_data, [config.TARGET_COLUMN])


def test_check_non_dataframe():
    """Test that non-DataFrame input raises TypeError."""

    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        Visualize._check([1, 2, 3], ["Score"])


def test_check_empty_dataframe():
    """Test that an empty DataFrame raises ValueError."""

    with pytest.raises(
        ValueError,
        match="Cannot plot an empty DataFrame",
    ):
        Visualize._check(pd.DataFrame(), ["Score"])


def test_check_missing_columns(valid_data):
    """Test that missing columns raise KeyError."""

    with pytest.raises(KeyError, match="Missing columns for plot"):
        Visualize._check(valid_data, ["Unknown_Column"])


def test_check_wrong_numeric_type(valid_data):
    """Test that non-numeric columns raise TypeError."""

    col = config.TARGET_COLUMN
    valid_data[col] = ["Low", "Medium", "High", "Low"]

    with pytest.raises(TypeError, match=f"Column '{col}' must be numeric"):
        Visualize._check(valid_data, [col])


# ----------------------------------
# Tests for plot_score_distribution()
# ----------------------------------

def test_plot_score_distribution_returns_figure(valid_data):
    """Test that score distribution returns a Matplotlib Figure."""

    fig = Visualize.plot_score_distribution(valid_data)

    try:
        assert isinstance(fig, plt.Figure)
        assert fig.axes[0].get_title() == "Exam Score Distribution"
        assert fig.axes[0].get_xlabel() == "Exam Score"
        assert fig.axes[0].get_ylabel() == "Number of Students"
    finally:
        plt.close(fig)


def test_plot_score_distribution_invalid_data():
    """Test that invalid input is rejected."""

    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        Visualize.plot_score_distribution([1, 2, 3])


# -------------------------
# Tests for plot_scatter()
# -------------------------

def test_plot_scatter_returns_figure(valid_data):
    """Test that scatter plot returns a Matplotlib Figure."""

    feature = "Hours_Studied"
    fig = Visualize.plot_scatter(valid_data, feature)

    try:
        assert isinstance(fig, plt.Figure)
        assert fig.axes[0].get_title() == "Hours Studied vs Exam Score"
        assert fig.axes[0].get_xlabel() == "Hours Studied"
        assert fig.axes[0].get_ylabel() == "Exam Score"
    finally:
        plt.close(fig)


def test_plot_scatter_missing_feature(valid_data):
    """Test that a missing feature raises KeyError."""

    with pytest.raises(KeyError, match="Missing columns for plot"):
        Visualize.plot_scatter(valid_data, "Unknown_Feature")


def test_plot_scatter_non_numeric_feature(valid_data):
    """Test that a non-numeric feature raises TypeError."""

    feature = "Hours_Studied"
    valid_data[feature] = ["Low", "Medium", "High", "Low"]

    with pytest.raises(TypeError, match=f"Column '{feature}' must be numeric"):
        Visualize.plot_scatter(valid_data, feature)


# --------------------------------
# Tests for plot_mean_by_category()
# --------------------------------

def test_plot_mean_by_category_returns_figure(valid_data):
    """Test that category mean plot returns a Matplotlib Figure."""

    category = "Motivation_Level"
    fig = Visualize.plot_mean_by_category(valid_data, category)

    try:
        assert isinstance(fig, plt.Figure)
        assert fig.axes[0].get_title() == "Average Exam Score by Motivation Level"
        assert fig.axes[0].get_xlabel() == "Motivation Level"
        assert fig.axes[0].get_ylabel() == "Average Exam Score"
    finally:
        plt.close(fig)


def test_plot_mean_by_category_non_dataframe():
    """Test that non-DataFrame input raises TypeError."""

    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        Visualize.plot_mean_by_category([1, 2, 3], "Category")


def test_plot_mean_by_category_empty_dataframe():
    """Test that an empty DataFrame raises ValueError."""

    with pytest.raises(
        ValueError,
        match="Cannot plot an empty DataFrame",
    ):
        Visualize.plot_mean_by_category(pd.DataFrame(), "Category")


def test_plot_mean_by_category_missing_column(valid_data):
    """Test that a missing category column raises KeyError."""

    with pytest.raises(KeyError, match="Missing columns for plot"):
        Visualize.plot_mean_by_category(valid_data, "Unknown_Category")


def test_plot_mean_by_category_wrong_target_type(valid_data):
    """Test that a non-numeric target raises TypeError."""

    target = config.TARGET_COLUMN
    valid_data[target] = ["Low", "Medium", "High", "Low"]

    with pytest.raises(TypeError, match=f"Column '{target}' must be numeric"):
        Visualize.plot_mean_by_category(valid_data, "Motivation_Level")


def test_plot_mean_by_category_wrong_category_type(valid_data):
    """Test that a numeric category column raises TypeError."""

    category = "Hours_Studied"

    with pytest.raises(
        TypeError,
        match=f"Column '{category}' must be categorical",
    ):
        Visualize.plot_mean_by_category(valid_data, category)


def test_plot_mean_by_category_string_dtype(valid_data):
    """Test that string dtype categorical columns are accepted."""

    category = "Motivation_Level"
    valid_data[category] = valid_data[category].astype("string")

    fig = Visualize.plot_mean_by_category(valid_data, category)

    try:
        assert isinstance(fig, plt.Figure)
        assert fig.axes[0].get_title() == (
            "Average Exam Score by Motivation Level"
        )
    finally:
        plt.close(fig)


# -----------------------------------
# Tests for plot_correlation_heatmap()
# -----------------------------------

def test_plot_correlation_heatmap_returns_figure(valid_data):
    """Test that correlation heatmap returns a Matplotlib Figure."""

    fig = Visualize.plot_correlation_heatmap(valid_data)

    try:
        assert isinstance(fig, plt.Figure)
        assert fig.axes[0].get_title() == "Correlation Between Numeric Features"
    finally:
        plt.close(fig)


def test_plot_correlation_heatmap_missing_column(valid_data):
    """Test that a missing numeric column raises KeyError."""

    col = list(config.NUMERIC_RANGES)[0]
    valid_data = valid_data.drop(columns=[col])

    with pytest.raises(KeyError, match="Missing columns for plot"):
        Visualize.plot_correlation_heatmap(valid_data)


# -------------------------
# Tests for save_figure()
# -------------------------

def test_save_figure_creates_file(temporary_visualizations_dir):
    """Test that a figure is saved successfully."""

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])

    path = Visualize.save_figure(fig, "test_chart.png")

    assert path.exists()
    assert path.parent == temporary_visualizations_dir
    assert path.name == "test_chart.png"
    assert path.stat().st_size > 0
    assert not plt.fignum_exists(fig.number)


def test_save_figure_none():
    """Test that saving None raises ValueError."""

    with pytest.raises(ValueError, match="Cannot save an empty figure"):
        Visualize.save_figure(None, "test.png")


@pytest.mark.parametrize("filename", ["", "   ", None, 123])
def test_save_figure_invalid_filename(filename):
    """Test that invalid filenames raise ValueError."""

    fig, _ = plt.subplots()

    with pytest.raises(
        ValueError,
        match="filename must be a non-empty string",
    ):
        Visualize.save_figure(fig, filename)

    plt.close(fig)


def test_save_figure_logs_success(
    temporary_visualizations_dir,
    caplog,
):
    """Test that saving a figure generates an INFO log."""

    fig, _ = plt.subplots()

    with caplog.at_level(logging.INFO, logger="Visualize"):
        path = Visualize.save_figure(fig, "logged_chart.png")

    assert path.exists()
    assert "Saved visualization: logged_chart.png" in caplog.text


# --------------------------------------
# Tests for generate_all_visualizations()
# --------------------------------------

def test_generate_all_visualizations(
    valid_data,
    temporary_visualizations_dir,
):
    """Test that all five project visualizations are generated and saved."""

    paths = Visualize.generate_all_visualizations(valid_data)

    expected_files = [
        "score_distribution.png",
        "hours_vs_score.png",
        "attendance_vs_score.png",
        "motivation_vs_score.png",
        "correlation_heatmap.png",
    ]

    assert len(paths) == 5
    assert [path.name for path in paths] == expected_files

    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0
        assert path.parent == temporary_visualizations_dir


def test_generate_all_visualizations_logs_success(
    valid_data,
    temporary_visualizations_dir,
    caplog,
):
    """Test that successful generation is logged."""

    with caplog.at_level(logging.INFO, logger="Visualize"):
        paths = Visualize.generate_all_visualizations(valid_data)

    assert len(paths) == 5
    assert "Generated 5 visualizations successfully" in caplog.text


def test_generate_all_visualizations_closes_figure_on_error(
    valid_data,
    temporary_visualizations_dir,
    monkeypatch,
):
    """Test that a figure is closed if saving fails."""

    created_figures = []

    def failing_save_figure(fig, filename):
        created_figures.append(fig)
        raise OSError("Simulated save failure")

    monkeypatch.setattr(
        Visualize,
        "save_figure",
        failing_save_figure,
    )

    with pytest.raises(OSError, match="Simulated save failure"):
        Visualize.generate_all_visualizations(valid_data)

    assert len(created_figures) == 1
    assert not plt.fignum_exists(created_figures[0].number)
