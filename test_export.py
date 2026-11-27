
"""Tests for Step 7 - Export."""

import joblib
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import config
from Export import export_data


@pytest.fixture
def sample_data():
    train_df = pd.DataFrame({
        "Hours_Studied": [5, 7, 8],
        config.TARGET_COLUMN: [65, 75, 85],
    })

    test_df = pd.DataFrame({
        "Hours_Studied": [4, 9],
        config.TARGET_COLUMN: [60, 90],
    })

    return train_df, test_df


@pytest.fixture
def fitted_preprocessor():
    preprocessor = StandardScaler()
    preprocessor.fit(pd.DataFrame({
        "Hours_Studied": [2, 4, 6],
    }))
    return preprocessor


def test_export_data_success(
    tmp_path, monkeypatch, sample_data, fitted_preprocessor
):
    train_df, test_df = sample_data

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")

    result = export_data(train_df, test_df, fitted_preprocessor)

    assert result["train_data"].exists()
    assert result["test_data"].exists()
    assert result["preprocessor"].exists()


def test_exported_training_data_content(
    tmp_path, monkeypatch, sample_data, fitted_preprocessor
):
    train_df, test_df = sample_data

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")

    result = export_data(train_df, test_df, fitted_preprocessor)

    exported_train = pd.read_csv(result["train_data"])

    pd.testing.assert_frame_equal(exported_train, train_df)


def test_exported_testing_data_content(
    tmp_path, monkeypatch, sample_data, fitted_preprocessor
):
    train_df, test_df = sample_data

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")

    result = export_data(train_df, test_df, fitted_preprocessor)

    exported_test = pd.read_csv(result["test_data"])

    pd.testing.assert_frame_equal(exported_test, test_df)


def test_exported_preprocessor_can_be_loaded(
    tmp_path, monkeypatch, sample_data, fitted_preprocessor
):
    train_df, test_df = sample_data

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")

    result = export_data(train_df, test_df, fitted_preprocessor)

    loaded_preprocessor = joblib.load(result["preprocessor"])

    assert hasattr(loaded_preprocessor, "feature_names_in_")


def test_export_creates_missing_directories(
    tmp_path, monkeypatch, sample_data, fitted_preprocessor
):
    train_df, test_df = sample_data

    processed_dir = tmp_path / "new_processed"
    models_dir = tmp_path / "new_models"

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)

    export_data(train_df, test_df, fitted_preprocessor)

    assert processed_dir.is_dir()
    assert models_dir.is_dir()


def test_export_rejects_non_dataframe_training_data(
    tmp_path, monkeypatch, sample_data, fitted_preprocessor
):
    _, test_df = sample_data

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")

    with pytest.raises(TypeError):
        export_data([], test_df, fitted_preprocessor)


def test_export_rejects_non_dataframe_testing_data(
    tmp_path, monkeypatch, sample_data, fitted_preprocessor
):
    train_df, _ = sample_data

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")

    with pytest.raises(TypeError):
        export_data(train_df, [], fitted_preprocessor)


def test_export_rejects_empty_training_data(
    tmp_path, monkeypatch, sample_data, fitted_preprocessor
):
    _, test_df = sample_data
    empty_df = pd.DataFrame(columns=[
        "Hours_Studied",
        config.TARGET_COLUMN,
    ])

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")

    with pytest.raises(ValueError, match="empty"):
        export_data(empty_df, test_df, fitted_preprocessor)


def test_export_rejects_empty_testing_data(
    tmp_path, monkeypatch, sample_data, fitted_preprocessor
):
    train_df, _ = sample_data
    empty_df = pd.DataFrame(columns=[
        "Hours_Studied",
        config.TARGET_COLUMN,
    ])

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")

    with pytest.raises(ValueError, match="empty"):
        export_data(train_df, empty_df, fitted_preprocessor)


def test_export_rejects_unfitted_preprocessor(
    tmp_path, monkeypatch, sample_data
):
    train_df, test_df = sample_data
    unfitted_preprocessor = StandardScaler()

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")

    with pytest.raises(ValueError, match="fitted"):
        export_data(train_df, test_df, unfitted_preprocessor)


def test_export_rejects_missing_target_in_training_data(
    tmp_path, monkeypatch, sample_data, fitted_preprocessor
):
    _, test_df = sample_data
    train_df = pd.DataFrame({"Hours_Studied": [5, 7]})

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")

    with pytest.raises(KeyError, match="training"):
        export_data(train_df, test_df, fitted_preprocessor)


def test_export_rejects_missing_target_in_testing_data(
    tmp_path, monkeypatch, sample_data, fitted_preprocessor
):
    train_df, _ = sample_data
    test_df = pd.DataFrame({"Hours_Studied": [4, 9]})

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")

    with pytest.raises(KeyError, match="testing"):
        export_data(train_df, test_df, fitted_preprocessor)

