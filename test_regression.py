
"""Tests for the Regression Model."""

import logging

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import GradientBoostingRegressor

import config
from Regression import train_regression_model


@pytest.fixture
def valid_train_data():
    """Create valid transformed training data."""

    n_rows = 30
    rng = np.random.default_rng(42)

    data = {}

    for i in range(40):
        data[f"feature_{i}"] = rng.normal(size=n_rows)

    data[config.TARGET_COLUMN] = rng.uniform(
        config.SCORE_MIN,
        config.SCORE_MAX,
        size=n_rows,
    )

    return pd.DataFrame(data)


# --------------------------------
# Tests for successful training
# --------------------------------

def test_train_regression_model_returns_model(
    valid_train_data,
    tmp_path,
):
    """Test that training returns a GradientBoostingRegressor."""

    model = train_regression_model(
        valid_train_data,
        tmp_path / "regression_model.joblib",
    )

    assert isinstance(model, GradientBoostingRegressor)


def test_train_regression_model_feature_count(
    valid_train_data,
    tmp_path,
):
    """Test that the model uses all input features."""

    model = train_regression_model(
        valid_train_data,
        tmp_path / "regression_model.joblib",
    )

    assert model.n_features_in_ == 40


def test_train_regression_model_does_not_use_target_as_feature(
    valid_train_data,
    tmp_path,
):
    """Test that Exam_Score is excluded from model inputs."""

    model = train_regression_model(
        valid_train_data,
        tmp_path / "regression_model.joblib",
    )

    assert model.n_features_in_ == len(valid_train_data.columns) - 1
    assert config.TARGET_COLUMN not in model.feature_names_in_


def test_train_regression_model_can_predict(
    valid_train_data,
    tmp_path,
):
    """Test that the trained model generates numeric predictions."""

    model = train_regression_model(
        valid_train_data,
        tmp_path / "regression_model.joblib",
    )

    X = valid_train_data.drop(columns=[config.TARGET_COLUMN])

    predictions = model.predict(X)

    assert len(predictions) == len(valid_train_data)
    assert np.issubdtype(predictions.dtype, np.number)
    assert np.isfinite(predictions).all()


def test_train_regression_model_predictions_within_score_range(
    valid_train_data,
    tmp_path,
):
    """Test that predictions stay within the training target range."""

    model = train_regression_model(
        valid_train_data,
        tmp_path / "regression_model.joblib",
    )

    X = valid_train_data.drop(columns=[config.TARGET_COLUMN])
    predictions = model.predict(X)

    assert predictions.min() >= valid_train_data[config.TARGET_COLUMN].min()
    assert predictions.max() <= valid_train_data[config.TARGET_COLUMN].max()


# --------------------------------
# Tests for model saving
# --------------------------------

def test_train_regression_model_saves_file(
    valid_train_data,
    tmp_path,
):
    """Test that the trained model is saved."""

    model_path = tmp_path / "regression_model.joblib"

    train_regression_model(valid_train_data, model_path)

    assert model_path.exists()


def test_train_regression_model_saved_model_can_be_loaded(
    valid_train_data,
    tmp_path,
):
    """Test that the saved model can be loaded using joblib."""

    model_path = tmp_path / "regression_model.joblib"

    train_regression_model(valid_train_data, model_path)

    loaded_model = joblib.load(model_path)

    assert isinstance(loaded_model, GradientBoostingRegressor)
    assert loaded_model.n_features_in_ == 40


def test_train_regression_model_creates_parent_directory(
    valid_train_data,
    tmp_path,
):
    """Test that missing output directories are created."""

    model_path = tmp_path / "new_models" / "regression_model.joblib"

    train_regression_model(valid_train_data, model_path)

    assert model_path.exists()


# --------------------------------
# Tests for input validation
# --------------------------------

def test_train_regression_model_rejects_non_dataframe(tmp_path):
    """Test that non-DataFrame input raises TypeError."""

    with pytest.raises(
        TypeError,
        match="train_df must be a pandas DataFrame",
    ):
        train_regression_model(
            [1, 2, 3],
            tmp_path / "model.joblib",
        )


def test_train_regression_model_rejects_empty_dataframe(tmp_path):
    """Test that empty training data raises ValueError."""

    with pytest.raises(
        ValueError,
        match="Training data cannot be empty",
    ):
        train_regression_model(
            pd.DataFrame(),
            tmp_path / "model.joblib",
        )


def test_train_regression_model_rejects_missing_target(
    valid_train_data,
    tmp_path,
):
    """Test that a missing target column raises KeyError."""

    invalid_data = valid_train_data.drop(
        columns=[config.TARGET_COLUMN]
    )

    with pytest.raises(
        KeyError,
        match="Target column",
    ):
        train_regression_model(
            invalid_data,
            tmp_path / "model.joblib",
        )


def test_train_regression_model_rejects_missing_target_values(
    valid_train_data,
    tmp_path,
):
    """Test that missing target values raise ValueError."""

    invalid_data = valid_train_data.copy()
    invalid_data.loc[0, config.TARGET_COLUMN] = np.nan

    with pytest.raises(
        ValueError,
        match="Target column contains missing values",
    ):
        train_regression_model(
            invalid_data,
            tmp_path / "model.joblib",
        )


def test_train_regression_model_rejects_non_numeric_target(
    valid_train_data,
    tmp_path,
):
    """Test that a non-numeric target raises TypeError."""

    invalid_data = valid_train_data.copy()
    invalid_data[config.TARGET_COLUMN] = "invalid"

    with pytest.raises(
        TypeError,
        match="Regression target must be numeric",
    ):
        train_regression_model(
            invalid_data,
            tmp_path / "model.joblib",
        )


def test_train_regression_model_rejects_missing_features(
    valid_train_data,
    tmp_path,
):
    """Test that missing feature values raise ValueError."""

    invalid_data = valid_train_data.copy()
    invalid_data.loc[0, "feature_0"] = np.nan

    with pytest.raises(
        ValueError,
        match="Input features contain missing values",
    ):
        train_regression_model(
            invalid_data,
            tmp_path / "model.joblib",
        )


def test_train_regression_model_rejects_non_numeric_features(
    valid_train_data,
    tmp_path,
):
    """Test that non-numeric features raise TypeError."""

    invalid_data = valid_train_data.copy()
    invalid_data["feature_0"] = "invalid"

    with pytest.raises(
        TypeError,
        match="All transformed features must be numeric",
    ):
        train_regression_model(
            invalid_data,
            tmp_path / "model.joblib",
        )


def test_train_regression_model_rejects_infinite_features(
    valid_train_data,
    tmp_path,
):
    """Test that infinite feature values raise ValueError."""

    invalid_data = valid_train_data.copy()
    invalid_data.loc[0, "feature_0"] = np.inf

    with pytest.raises(
        ValueError,
        match="Input features contain infinite values",
    ):
        train_regression_model(
            invalid_data,
            tmp_path / "model.joblib",
        )


def test_train_regression_model_rejects_infinite_target(
    valid_train_data,
    tmp_path,
):
    """Test that infinite target values raise ValueError."""

    invalid_data = valid_train_data.copy()
    invalid_data.loc[0, config.TARGET_COLUMN] = np.inf

    with pytest.raises(
        ValueError,
        match="Target contains infinite values",
    ):
        train_regression_model(
            invalid_data,
            tmp_path / "model.joblib",
        )


# --------------------------------
# Tests for reproducibility
# --------------------------------

def test_train_regression_model_is_reproducible(
    valid_train_data,
    tmp_path,
):
    """Test that the same data produces reproducible predictions."""

    model_1 = train_regression_model(
        valid_train_data,
        tmp_path / "model_1.joblib",
    )

    model_2 = train_regression_model(
        valid_train_data,
        tmp_path / "model_2.joblib",
    )

    X = valid_train_data.drop(columns=[config.TARGET_COLUMN])

    predictions_1 = model_1.predict(X)
    predictions_2 = model_2.predict(X)

    np.testing.assert_allclose(
        predictions_1,
        predictions_2,
    )


# --------------------------------
# Tests for logging
# --------------------------------

def test_train_regression_model_logs_success(
    valid_train_data,
    tmp_path,
    caplog,
):
    """Test that successful training generates INFO logs."""

    with caplog.at_level(logging.INFO, logger="Regression"):
        train_regression_model(
            valid_train_data,
            tmp_path / "regression_model.joblib",
        )

    assert "Regression model trained successfully" in caplog.text
    assert "Regression model saved to" in caplog.text


# --------------------------------
# Tests for loading training data
# --------------------------------

def test_train_regression_model_loads_default_training_file(
    valid_train_data,
    tmp_path,
    monkeypatch,
):
    """Test that training data is loaded from the configured directory."""

    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"

    processed_dir.mkdir()

    valid_train_data.to_csv(
        processed_dir / "train.csv",
        index=False,
    )

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)

    model = train_regression_model()

    assert isinstance(model, GradientBoostingRegressor)
    assert (models_dir / "regression_model.joblib").exists()


def test_train_regression_model_missing_default_file(
    tmp_path,
    monkeypatch,
):
    """Test that a missing default training file raises FileNotFoundError."""

    processed_dir = tmp_path / "processed"
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", processed_dir)

    with pytest.raises(
        FileNotFoundError,
        match="Training data file not found",
    ):
        train_regression_model()


def test_train_regression_model_invalid_csv(
    tmp_path,
    monkeypatch,
):
    """Test that an unreadable CSV raises ValueError."""

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    train_path = processed_dir / "train.csv"
    train_path.write_text('"unfinished', encoding="utf-8")

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", processed_dir)

    with pytest.raises(
        ValueError,
        match="Could not read the training data",
    ):
        train_regression_model()
