
"""Tests for Step 8 - Summary."""

import json

import pandas as pd
import pytest

import config
from Summary import generate_summary, save_summary


@pytest.fixture
def sample_data():
    data = {}

    for column, (minimum, maximum) in config.NUMERIC_RANGES.items():
        data[column] = [
            minimum + 1,
            minimum + 2,
            minimum + 3,
            minimum + 4,
        ]

    for column, categories in config.CATEGORICAL_VALUES.items():
        data[column] = [
            categories[0],
            categories[1] if len(categories) > 1 else categories[0],
            categories[0],
            categories[1] if len(categories) > 1 else categories[0],
        ]

    data[config.TARGET_COLUMN] = [55, 60, 80, 100]

    return pd.DataFrame(data)


def test_generate_summary_success(sample_data):
    summary = generate_summary(sample_data)

    assert isinstance(summary, dict)
    assert "dataset" in summary
    assert "numeric_summary" in summary
    assert "categorical_summary" in summary
    assert "performance_level_distribution" in summary


def test_summary_dataset_information(sample_data):
    summary = generate_summary(sample_data)

    assert summary["dataset"]["rows"] == 4
    assert summary["dataset"]["columns"] == sample_data.shape[1]
    assert summary["dataset"]["column_names"] == list(sample_data.columns)


def test_summary_numeric_statistics(sample_data):
    summary = generate_summary(sample_data)

    target_summary = summary["numeric_summary"][config.TARGET_COLUMN]

    assert target_summary["mean"] == pytest.approx(73.75)
    assert target_summary["median"] == pytest.approx(70)
    assert target_summary["min"] == 55
    assert target_summary["max"] == 100


def test_summary_performance_level_distribution(sample_data):
    summary = generate_summary(sample_data)

    distribution = summary["performance_level_distribution"]

    assert distribution == {
        "Low": 2,
        "Medium": 0,
        "High": 2,
    }


def test_summary_performance_level_boundaries(sample_data):
    sample_data[config.TARGET_COLUMN] = [59, 60, 79, 80]

    summary = generate_summary(sample_data)

    assert summary["performance_level_distribution"] == {
        "Low": 2,
        "Medium": 1,
        "High": 1,
    }


def test_summary_categorical_counts(sample_data):
    summary = generate_summary(sample_data)

    for column in config.CATEGORICAL_VALUES:
        assert column in summary["categorical_summary"]
        assert sum(
            summary["categorical_summary"][column].values()
        ) == len(sample_data)


def test_summary_missing_values_count(sample_data):
    sample_data.loc[0, config.TARGET_COLUMN] = None

    summary = generate_summary(sample_data)

    assert summary["dataset"]["missing_values"][config.TARGET_COLUMN] == 1

    assert summary["performance_level_distribution"] == {
        "Low": 1,
        "Medium": 0,
        "High": 2,
    }


def test_summary_duplicate_rows_count(sample_data):
    duplicated_data = pd.concat(
        [sample_data, sample_data.iloc[[0]]],
        ignore_index=True,
    )

    summary = generate_summary(duplicated_data)

    assert summary["dataset"]["duplicate_rows"] == 1


def test_summary_rejects_non_dataframe():
    with pytest.raises(TypeError):
        generate_summary([])


def test_summary_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty"):
        generate_summary(pd.DataFrame())


def test_summary_rejects_missing_required_columns(sample_data):
    sample_data = sample_data.drop(columns=[config.TARGET_COLUMN])

    with pytest.raises(KeyError, match="Missing required columns"):
        generate_summary(sample_data)


def test_summary_rejects_non_numeric_target(sample_data):
    sample_data[config.TARGET_COLUMN] = ["Low", "Medium", "High", "High"]

    with pytest.raises(TypeError, match="must be numeric"):
        generate_summary(sample_data)


def test_summary_rejects_non_numeric_feature(sample_data):
    column = list(config.NUMERIC_RANGES)[0]
    sample_data[column] = ["invalid"] * len(sample_data)

    with pytest.raises(TypeError, match="must be numeric"):
        generate_summary(sample_data)


def test_save_summary_success(tmp_path, monkeypatch, sample_data):
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path)

    summary = generate_summary(sample_data)
    output_path = save_summary(summary)

    assert output_path.exists()
    assert output_path.name == "data_summary.json"

    with output_path.open("r", encoding="utf-8") as file:
        saved_summary = json.load(file)

    assert saved_summary == summary


def test_save_summary_creates_directory(tmp_path, monkeypatch, sample_data):
    output_dir = tmp_path / "new_processed"

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", output_dir)

    summary = generate_summary(sample_data)
    output_path = save_summary(summary)

    assert output_dir.is_dir()
    assert output_path.exists()


def test_save_summary_rejects_empty_dictionary(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path)

    with pytest.raises(ValueError, match="non-empty dictionary"):
        save_summary({})


def test_save_summary_rejects_non_dictionary(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path)

    with pytest.raises(ValueError, match="non-empty dictionary"):
        save_summary([])


def test_save_summary_rejects_non_serializable_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path)

    invalid_summary = {"invalid": object()}

    with pytest.raises(TypeError):
        save_summary(invalid_summary)
