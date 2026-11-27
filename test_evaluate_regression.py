
"""Tests for the Regression Evaluation module."""

import logging

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import GradientBoostingRegressor

import config
from Evaluate_Regression import evaluate_regression_model


@pytest.fixture
def train_df():
    """Create sample training data."""
    return pd.DataFrame({
        "Feature_1": np.arange(20, dtype=float),
        "Feature_2": np.arange(20, dtype=float) * 2,
        config.TARGET_COLUMN: np.arange(20, dtype=float) + 50,
    })


@pytest.fixture
def test_df():
    """Create sample test data."""
    return pd.DataFrame({
        "Feature_1": np.arange(10, dtype=float),
        "Feature_2": np.arange(10, dtype=float) * 2,
        config.TARGET_COLUMN: np.arange(10, dtype=float) + 52,
    })


@pytest.fixture
def saved_model(tmp_path):
    """Create and save a fitted regression model."""
    X_train = pd.DataFrame({
        "Feature_1": np.arange(20, dtype=float),
        "Feature_2": np.arange(20, dtype=float) * 2,
    })

    y_train = np.arange(20, dtype=float) + 50

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=config.RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    model_path = tmp_path / "regression_model.joblib"
    joblib.dump(model, model_path)

    return model_path


@pytest.fixture
def evaluation_results(train_df, test_df, saved_model):
    """Run evaluation using sample data."""
    return evaluate_regression_model(
        test_df=test_df,
        train_df=train_df,
        model_path=saved_model,
    )


# --------------------------------------------------
# Basic evaluation tests
# --------------------------------------------------

def test_evaluation_returns_dictionary(evaluation_results):
    """Check that evaluation returns a dictionary."""
    assert isinstance(evaluation_results, dict)


def test_evaluation_contains_model_and_baseline(evaluation_results):
    """Check that both model and baseline results are returned."""
    assert "model" in evaluation_results
    assert "baseline" in evaluation_results


def test_evaluation_contains_all_metrics(evaluation_results):
    """Check that all required regression metrics are returned."""
    expected_metrics = {"MAE", "RMSE", "R2"}

    assert set(evaluation_results["model"].keys()) == expected_metrics
    assert set(evaluation_results["baseline"].keys()) == expected_metrics


def test_evaluation_metrics_are_numeric(evaluation_results):
    """Check that all metric values are numeric."""
    for group in evaluation_results.values():
        for value in group.values():
            assert isinstance(value, float)


def test_evaluation_metrics_are_finite(evaluation_results):
    """Check that all metric values are finite."""
    for group in evaluation_results.values():
        for value in group.values():
            assert np.isfinite(value)


def test_evaluation_errors_are_non_negative(evaluation_results):
    """Check that MAE and RMSE cannot be negative."""
    for group in evaluation_results.values():
        assert group["MAE"] >= 0
        assert group["RMSE"] >= 0


def test_baseline_is_fitted_using_training_mean(
    evaluation_results,
    train_df,
    test_df,
):
    """Check that the baseline predicts the training target mean."""
    training_mean = train_df[config.TARGET_COLUMN].mean()

    expected_mae = np.mean(
        np.abs(test_df[config.TARGET_COLUMN] - training_mean)
    )

    assert evaluation_results["baseline"]["MAE"] == pytest.approx(
        expected_mae
    )


def test_model_and_baseline_use_same_test_data(evaluation_results):
    """Check that both evaluation groups contain the same metrics."""
    assert set(evaluation_results["model"]) == set(
        evaluation_results["baseline"]
    )


# --------------------------------------------------
# Input validation tests
# --------------------------------------------------

def test_rejects_non_dataframe_test_data(
    train_df,
    saved_model,
):
    """Reject test data that is not a DataFrame."""
    with pytest.raises(TypeError, match="test_df"):
        evaluate_regression_model(
            test_df=[1, 2, 3],
            train_df=train_df,
            model_path=saved_model,
        )


def test_rejects_non_dataframe_training_data(
    test_df,
    saved_model,
):
    """Reject training data that is not a DataFrame."""
    with pytest.raises(TypeError, match="train_df"):
        evaluate_regression_model(
            test_df=test_df,
            train_df=[1, 2, 3],
            model_path=saved_model,
        )


def test_rejects_empty_test_data(
    train_df,
    saved_model,
):
    """Reject an empty test DataFrame."""
    with pytest.raises(ValueError, match="Test data cannot be empty"):
        evaluate_regression_model(
            test_df=pd.DataFrame(),
            train_df=train_df,
            model_path=saved_model,
        )


def test_rejects_empty_training_data(
    test_df,
    saved_model,
):
    """Reject an empty training DataFrame."""
    with pytest.raises(ValueError, match="Training data cannot be empty"):
        evaluate_regression_model(
            test_df=test_df,
            train_df=pd.DataFrame(),
            model_path=saved_model,
        )


def test_rejects_missing_test_target(
    train_df,
    test_df,
    saved_model,
):
    """Reject test data without the target column."""
    invalid_df = test_df.drop(columns=[config.TARGET_COLUMN])

    with pytest.raises(KeyError, match="missing from test data"):
        evaluate_regression_model(
            test_df=invalid_df,
            train_df=train_df,
            model_path=saved_model,
        )


def test_rejects_missing_training_target(
    train_df,
    test_df,
    saved_model,
):
    """Reject training data without the target column."""
    invalid_df = train_df.drop(columns=[config.TARGET_COLUMN])

    with pytest.raises(KeyError, match="missing from training data"):
        evaluate_regression_model(
            test_df=test_df,
            train_df=invalid_df,
            model_path=saved_model,
        )


def test_rejects_missing_test_target_values(
    train_df,
    test_df,
    saved_model,
):
    """Reject missing values in the test target."""
    invalid_df = test_df.copy()
    invalid_df.loc[0, config.TARGET_COLUMN] = np.nan

    with pytest.raises(ValueError, match="Test target contains missing"):
        evaluate_regression_model(
            test_df=invalid_df,
            train_df=train_df,
            model_path=saved_model,
        )


def test_rejects_missing_training_target_values(
    train_df,
    test_df,
    saved_model,
):
    """Reject missing values in the training target."""
    invalid_df = train_df.copy()
    invalid_df.loc[0, config.TARGET_COLUMN] = np.nan

    with pytest.raises(ValueError, match="Training target contains missing"):
        evaluate_regression_model(
            test_df=test_df,
            train_df=invalid_df,
            model_path=saved_model,
        )


def test_rejects_non_numeric_test_target(
    train_df,
    test_df,
    saved_model,
):
    """Reject a non-numeric test target."""
    invalid_df = test_df.copy()
    invalid_df[config.TARGET_COLUMN] = "invalid"

    with pytest.raises(TypeError, match="Test target must be numeric"):
        evaluate_regression_model(
            test_df=invalid_df,
            train_df=train_df,
            model_path=saved_model,
        )


def test_rejects_non_numeric_training_target(
    train_df,
    test_df,
    saved_model,
):
    """Reject a non-numeric training target."""
    invalid_df = train_df.copy()
    invalid_df[config.TARGET_COLUMN] = "invalid"

    with pytest.raises(TypeError, match="Training target must be numeric"):
        evaluate_regression_model(
            test_df=test_df,
            train_df=invalid_df,
            model_path=saved_model,
        )


def test_rejects_infinite_test_target(
    train_df,
    test_df,
    saved_model,
):
    """Reject infinite values in the test target."""
    invalid_df = test_df.copy()
    invalid_df.loc[0, config.TARGET_COLUMN] = np.inf

    with pytest.raises(ValueError, match="Test target contains infinite"):
        evaluate_regression_model(
            test_df=invalid_df,
            train_df=train_df,
            model_path=saved_model,
        )


def test_rejects_infinite_training_target(
    train_df,
    test_df,
    saved_model,
):
    """Reject infinite values in the training target."""
    invalid_df = train_df.copy()
    invalid_df.loc[0, config.TARGET_COLUMN] = np.inf

    with pytest.raises(ValueError, match="Training target contains infinite"):
        evaluate_regression_model(
            test_df=test_df,
            train_df=invalid_df,
            model_path=saved_model,
        )


def test_rejects_missing_test_features(
    train_df,
    test_df,
    saved_model,
):
    """Reject missing values in test features."""
    invalid_df = test_df.copy()
    invalid_df.loc[0, "Feature_1"] = np.nan

    with pytest.raises(ValueError, match="Test features contain missing"):
        evaluate_regression_model(
            test_df=invalid_df,
            train_df=train_df,
            model_path=saved_model,
        )


def test_rejects_non_numeric_test_features(
    train_df,
    test_df,
    saved_model,
):
    """Reject non-numeric test features."""
    invalid_df = test_df.copy()
    invalid_df["Feature_1"] = "invalid"

    with pytest.raises(TypeError, match="All test features must be numeric"):
        evaluate_regression_model(
            test_df=invalid_df,
            train_df=train_df,
            model_path=saved_model,
        )


def test_rejects_infinite_test_features(
    train_df,
    test_df,
    saved_model,
):
    """Reject infinite values in test features."""
    invalid_df = test_df.copy()
    invalid_df.loc[0, "Feature_1"] = np.inf

    with pytest.raises(ValueError, match="Test features contain infinite"):
        evaluate_regression_model(
            test_df=invalid_df,
            train_df=train_df,
            model_path=saved_model,
        )


def test_rejects_empty_test_features(
    train_df,
    saved_model,
):
    """Reject test data containing only the target column."""
    invalid_df = pd.DataFrame({
        config.TARGET_COLUMN: [60.0, 70.0, 80.0]
    })

    with pytest.raises(ValueError, match="No test input features"):
        evaluate_regression_model(
            test_df=invalid_df,
            train_df=train_df,
            model_path=saved_model,
        )


# --------------------------------------------------
# Model file tests
# --------------------------------------------------

def test_rejects_missing_model_file(
    train_df,
    test_df,
    tmp_path,
):
    """Reject a model path that does not exist."""
    missing_path = tmp_path / "missing_model.joblib"

    with pytest.raises(FileNotFoundError, match="Regression model not found"):
        evaluate_regression_model(
            test_df=test_df,
            train_df=train_df,
            model_path=missing_path,
        )


def test_rejects_corrupted_model_file(
    train_df,
    test_df,
    tmp_path,
):
    """Reject a corrupted model file."""
    corrupted_path = tmp_path / "corrupted_model.joblib"
    corrupted_path.write_text("invalid model content")

    with pytest.raises(ValueError, match="Could not load the regression model"):
        evaluate_regression_model(
            test_df=test_df,
            train_df=train_df,
            model_path=corrupted_path,
        )


def test_rejects_model_with_wrong_feature_count(
    train_df,
    test_df,
    tmp_path,
):
    """Reject a model expecting a different number of features."""
    X_train = pd.DataFrame({
        "Feature_1": np.arange(20, dtype=float),
    })
    y_train = np.arange(20, dtype=float) + 50

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    model_path = tmp_path / "wrong_features.joblib"
    joblib.dump(model, model_path)

    with pytest.raises(
        ValueError,
        match="Test feature count does not match",
    ):
        evaluate_regression_model(
            test_df=test_df,
            train_df=train_df,
            model_path=model_path,
        )


def test_rejects_model_with_wrong_feature_names(
    train_df,
    test_df,
    tmp_path,
):
    """Reject a model with different feature names."""
    X_train = pd.DataFrame({
        "Other_Feature_1": np.arange(20, dtype=float),
        "Other_Feature_2": np.arange(20, dtype=float) * 2,
    })
    y_train = np.arange(20, dtype=float) + 50

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    model_path = tmp_path / "wrong_names.joblib"
    joblib.dump(model, model_path)

    with pytest.raises(
        ValueError,
        match="Test feature names or order",
    ):
        evaluate_regression_model(
            test_df=test_df,
            train_df=train_df,
            model_path=model_path,
        )


# --------------------------------------------------
# Default file loading tests
# --------------------------------------------------

def test_loads_default_training_and_test_files(
    train_df,
    test_df,
    saved_model,
    monkeypatch,
    tmp_path,
):
    """Load train.csv and test.csv from the configured directory."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    train_df.to_csv(processed_dir / "train.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    monkeypatch.setattr(
        config,
        "PROCESSED_DATA_DIR",
        processed_dir,
    )

    results = evaluate_regression_model(
        model_path=saved_model,
    )

    assert "model" in results
    assert "baseline" in results


def test_rejects_missing_default_test_file(
    train_df,
    saved_model,
    monkeypatch,
    tmp_path,
):
    """Reject a missing default test.csv file."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    monkeypatch.setattr(
        config,
        "PROCESSED_DATA_DIR",
        processed_dir,
    )

    with pytest.raises(FileNotFoundError, match="Test data file not found"):
        evaluate_regression_model(
            train_df=train_df,
            model_path=saved_model,
        )


def test_rejects_missing_default_training_file(
    test_df,
    saved_model,
    monkeypatch,
    tmp_path,
):
    """Reject a missing default train.csv file."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    test_df.to_csv(processed_dir / "test.csv", index=False)

    monkeypatch.setattr(
        config,
        "PROCESSED_DATA_DIR",
        processed_dir,
    )

    with pytest.raises(FileNotFoundError, match="Training data file not found"):
        evaluate_regression_model(
            model_path=saved_model,
        )


def test_rejects_invalid_test_csv(
    train_df,
    saved_model,
    monkeypatch,
    tmp_path,
):
    """Reject an invalid test CSV file."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    (processed_dir / "test.csv").write_text('"unfinished')
    train_df.to_csv(processed_dir / "train.csv", index=False)

    monkeypatch.setattr(
        config,
        "PROCESSED_DATA_DIR",
        processed_dir,
    )

    with pytest.raises(ValueError, match="Could not read the test data"):
        evaluate_regression_model(
            model_path=saved_model,
        )


def test_rejects_invalid_training_csv(
    test_df,
    saved_model,
    monkeypatch,
    tmp_path,
):
    """Reject an invalid training CSV file."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    test_df.to_csv(processed_dir / "test.csv", index=False)
    (processed_dir / "train.csv").write_text('"unfinished')

    monkeypatch.setattr(
        config,
        "PROCESSED_DATA_DIR",
        processed_dir,
    )

    with pytest.raises(
        ValueError,
        match="Could not read the training data",
    ):
        evaluate_regression_model(
            model_path=saved_model,
        )


# --------------------------------------------------
# Logging test
# --------------------------------------------------

def test_evaluation_logs_success(
    train_df,
    test_df,
    saved_model,
    caplog,
):
    """Check that successful evaluation is logged."""

    with caplog.at_level(logging.INFO):
        evaluate_regression_model(
            test_df=test_df,
            train_df=train_df,
            model_path=saved_model,
        )

    assert "Regression evaluation completed" in caplog.text
