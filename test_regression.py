

import joblib
import numpy as np

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from Regression import (
    load_training_data,
    train_linear_regression,
    train_random_forest,
    compare_models,
    save_model,
)


def test_load_training_data():
    """Test loading and validating training data."""

    X, y = load_training_data()

    assert not X.empty
    assert len(X) == len(y)
    assert "Exam_Score" not in X.columns
    assert not X.isna().any().any()
    assert not y.isna().any()


def test_linear_regression():
    """Test Linear Regression training and predictions."""

    X, y = load_training_data()

    model = train_linear_regression(X.head(100), y.head(100))

    assert isinstance(model, LinearRegression)

    predictions = model.predict(X.head(5))

    assert len(predictions) == 5
    assert np.isfinite(predictions).all()


def test_random_forest():
    """Test Random Forest training and predictions."""

    X, y = load_training_data()

    model = train_random_forest(X.head(100), y.head(100))

    assert isinstance(model, RandomForestRegressor)

    predictions = model.predict(X.head(5))

    assert len(predictions) == 5
    assert np.isfinite(predictions).all()


def test_compare_models():
    """Test model comparison and evaluation metrics."""

    results = compare_models()

    assert "Linear Regression" in results
    assert "Random Forest Regressor" in results

    for metrics in results.values():
        assert set(metrics.keys()) == {"MAE", "RMSE", "R2"}

        assert np.isfinite(metrics["MAE"])
        assert np.isfinite(metrics["RMSE"])
        assert np.isfinite(metrics["R2"])

        assert metrics["MAE"] >= 0
        assert metrics["RMSE"] >= 0
        assert metrics["R2"] <= 1


def test_save_model(tmp_path, monkeypatch):
    """Test saving and reloading a trained model."""

    X, y = load_training_data()

    model = train_linear_regression(X.head(100), y.head(100))

    # Redirect model saving to a temporary directory
    original_path = Path
    monkeypatch.setattr(
        "Regression.Path",
        lambda _: tmp_path
    )

    model_path = save_model(model, "test_model.joblib")

    assert model_path.exists()
    assert model_path.name == "test_model.joblib"

    loaded_model = joblib.load(model_path)

    original_predictions = model.predict(X.head(5))
    loaded_predictions = loaded_model.predict(X.head(5))

    assert np.allclose(
        original_predictions,
        loaded_predictions
    )
