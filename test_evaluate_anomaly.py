
"""Tests for the Anomaly Detection evaluation module."""

import joblib
import numpy as np
import pandas as pd
import pytest

import config
from sklearn.ensemble import IsolationForest

from Anomaly_Detection import train_anomaly_model
from Evaluate_Anomaly import (
    prepare_evaluation_features,
    evaluate_anomaly_model,
    print_results,
)


@pytest.fixture
def sample_train_data():
    """Create valid training data."""

    rng = np.random.default_rng(42)

    return pd.DataFrame({
        "Hours_Studied": rng.normal(20, 5, 100),
        "Attendance": rng.normal(80, 10, 100),
        "Sleep_Hours": rng.normal(7, 1, 100),
        "Previous_Scores": rng.normal(70, 10, 100),
        config.TARGET_COLUMN: rng.normal(67, 4, 100),
    })


@pytest.fixture
def sample_test_data():
    """Create valid testing data with matching features."""

    rng = np.random.default_rng(100)

    return pd.DataFrame({
        "Hours_Studied": rng.normal(20, 5, 30),
        "Attendance": rng.normal(80, 10, 30),
        "Sleep_Hours": rng.normal(7, 1, 30),
        "Previous_Scores": rng.normal(70, 10, 30),
        config.TARGET_COLUMN: rng.normal(67, 4, 30),
    })


@pytest.fixture
def trained_model(sample_train_data, tmp_path):
    """Train and save a model for evaluation tests."""

    model_path = tmp_path / "anomaly_model.joblib"

    train_anomaly_model(
        sample_train_data,
        model_path=model_path,
    )

    return model_path


# ---------------------------------------------------------
# prepare_evaluation_features tests
# ---------------------------------------------------------

def test_prepare_evaluation_features_returns_dataframe(
    sample_test_data,
    trained_model,
):
    model = joblib.load(trained_model)

    result = prepare_evaluation_features(sample_test_data, model)

    assert isinstance(result, pd.DataFrame)


def test_prepare_evaluation_features_excludes_target(
    sample_test_data,
    trained_model,
):
    model = joblib.load(trained_model)

    result = prepare_evaluation_features(sample_test_data, model)

    assert config.TARGET_COLUMN not in result.columns


def test_prepare_evaluation_features_preserves_rows(
    sample_test_data,
    trained_model,
):
    model = joblib.load(trained_model)

    result = prepare_evaluation_features(sample_test_data, model)

    assert len(result) == len(sample_test_data)


def test_prepare_evaluation_features_preserves_feature_order(
    sample_test_data,
    trained_model,
):
    model = joblib.load(trained_model)

    result = prepare_evaluation_features(sample_test_data, model)

    expected = [
        col for col in sample_test_data.columns
        if col != config.TARGET_COLUMN
    ]

    assert list(result.columns) == expected


def test_prepare_evaluation_features_rejects_non_dataframe(
    trained_model,
):
    model = joblib.load(trained_model)

    with pytest.raises(TypeError, match="pandas DataFrame"):
        prepare_evaluation_features([1, 2, 3], model)


def test_prepare_evaluation_features_rejects_empty_dataframe(
    trained_model,
):
    model = joblib.load(trained_model)

    empty_df = pd.DataFrame(
        columns=[
            "Hours_Studied",
            config.TARGET_COLUMN,
        ]
    )

    with pytest.raises(ValueError, match="empty"):
        prepare_evaluation_features(empty_df, model)


def test_prepare_evaluation_features_rejects_missing_target(
    trained_model,
):
    model = joblib.load(trained_model)

    df = pd.DataFrame({
        "Hours_Studied": [10, 20, 30],
        "Attendance": [70, 80, 90],
        "Sleep_Hours": [6, 7, 8],
        "Previous_Scores": [60, 70, 80],
    })

    with pytest.raises(KeyError, match="Missing target"):
        prepare_evaluation_features(df, model)


def test_prepare_evaluation_features_rejects_duplicate_columns(
    trained_model,
):
    model = joblib.load(trained_model)

    df = pd.DataFrame(
        [
            [10, 20, 70, 6, 60],
            [15, 25, 80, 7, 70],
        ],
        columns=[
            "Hours_Studied",
            "Hours_Studied",
            "Attendance",
            "Sleep_Hours",
            config.TARGET_COLUMN,
        ],
    )

    with pytest.raises(ValueError, match="duplicate"):
        prepare_evaluation_features(df, model)


def test_prepare_evaluation_features_rejects_missing_values(
    sample_test_data,
    trained_model,
):
    model = joblib.load(trained_model)

    df = sample_test_data.copy()
    df.loc[0, "Hours_Studied"] = np.nan

    with pytest.raises(ValueError, match="Missing values"):
        prepare_evaluation_features(df, model)


def test_prepare_evaluation_features_rejects_nonnumeric_features(
    sample_test_data,
    trained_model,
):
    model = joblib.load(trained_model)

    df = sample_test_data.copy()
    df["Hours_Studied"] = df["Hours_Studied"].astype(str)

    with pytest.raises(TypeError, match="must contain numeric"):
        prepare_evaluation_features(df, model)


@pytest.mark.parametrize("invalid_value", [np.inf, -np.inf])
def test_prepare_evaluation_features_rejects_infinite_values(
    sample_test_data,
    trained_model,
    invalid_value,
):
    model = joblib.load(trained_model)

    df = sample_test_data.copy()
    df.loc[0, "Hours_Studied"] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        prepare_evaluation_features(df, model)


def test_prepare_evaluation_features_rejects_no_features(
    trained_model,
):
    model = joblib.load(trained_model)

    df = pd.DataFrame({
        config.TARGET_COLUMN: [60, 70, 80]
    })

    with pytest.raises(ValueError, match="No features"):
        prepare_evaluation_features(df, model)


def test_prepare_evaluation_features_rejects_wrong_feature_order(
    sample_test_data,
    trained_model,
):
    model = joblib.load(trained_model)

    df = sample_test_data.copy()

    feature_columns = [
        col for col in df.columns
        if col != config.TARGET_COLUMN
    ]

    reordered = feature_columns[::-1] + [config.TARGET_COLUMN]
    df = df[reordered]

    with pytest.raises(ValueError, match="feature"):
        prepare_evaluation_features(df, model)


def test_prepare_evaluation_features_rejects_missing_feature(
    sample_test_data,
    trained_model,
):
    model = joblib.load(trained_model)

    df = sample_test_data.drop(columns=["Attendance"])

    with pytest.raises(ValueError, match="feature"):
        prepare_evaluation_features(df, model)


# ---------------------------------------------------------
# evaluate_anomaly_model tests
# ---------------------------------------------------------

def test_evaluate_anomaly_model_returns_expected_keys(
    sample_test_data,
    trained_model,
):
    result = evaluate_anomaly_model(
        sample_test_data,
        model_path=trained_model,
    )

    expected_keys = {
        "results",
        "anomaly_count",
        "normal_count",
        "anomaly_percentage",
        "mean_anomaly_score",
        "score_summary",
        "feature_comparison",
    }

    assert set(result.keys()) == expected_keys


def test_evaluate_anomaly_model_returns_dataframe(
    sample_test_data,
    trained_model,
):
    result = evaluate_anomaly_model(
        sample_test_data,
        model_path=trained_model,
    )

    assert isinstance(result["results"], pd.DataFrame)


def test_evaluate_anomaly_model_returns_expected_columns(
    sample_test_data,
    trained_model,
):
    result = evaluate_anomaly_model(
        sample_test_data,
        model_path=trained_model,
    )

    expected_columns = {
        "Prediction",
        "Anomaly_Score",
        config.TARGET_COLUMN,
        "Status",
    }

    assert expected_columns.issubset(result["results"].columns)


def test_evaluate_anomaly_model_preserves_test_rows(
    sample_test_data,
    trained_model,
):
    result = evaluate_anomaly_model(
        sample_test_data,
        model_path=trained_model,
    )

    assert len(result["results"]) == len(sample_test_data)


def test_evaluate_anomaly_model_returns_valid_predictions(
    sample_test_data,
    trained_model,
):
    result = evaluate_anomaly_model(
        sample_test_data,
        model_path=trained_model,
    )

    predictions = result["results"]["Prediction"]

    assert set(predictions.unique()).issubset({-1, 1})


def test_evaluate_anomaly_model_returns_valid_statuses(
    sample_test_data,
    trained_model,
):
    result = evaluate_anomaly_model(
        sample_test_data,
        model_path=trained_model,
    )

    statuses = result["results"]["Status"]

    assert set(statuses.unique()).issubset({"Anomaly", "Normal"})


def test_evaluate_anomaly_model_counts_are_consistent(
    sample_test_data,
    trained_model,
):
    result = evaluate_anomaly_model(
        sample_test_data,
        model_path=trained_model,
    )

    assert result["anomaly_count"] == (
        result["results"]["Prediction"] == -1
    ).sum()

    assert result["normal_count"] == (
        result["results"]["Prediction"] == 1
    ).sum()

    assert (
        result["anomaly_count"] + result["normal_count"]
        == len(sample_test_data)
    )


def test_evaluate_anomaly_model_percentage_is_valid(
    sample_test_data,
    trained_model,
):
    result = evaluate_anomaly_model(
        sample_test_data,
        model_path=trained_model,
    )

    assert 0 <= result["anomaly_percentage"] <= 100


def test_evaluate_anomaly_model_scores_are_finite(
    sample_test_data,
    trained_model,
):
    result = evaluate_anomaly_model(
        sample_test_data,
        model_path=trained_model,
    )

    scores = result["results"]["Anomaly_Score"]

    assert np.isfinite(scores).all()
    assert np.isfinite(result["mean_anomaly_score"])


def test_evaluate_anomaly_model_score_summary_is_dataframe(
    sample_test_data,
    trained_model,
):
    result = evaluate_anomaly_model(
        sample_test_data,
        model_path=trained_model,
    )

    assert isinstance(result["score_summary"], pd.DataFrame)


def test_evaluate_anomaly_model_feature_comparison_is_dataframe(
    sample_test_data,
    trained_model,
):
    result = evaluate_anomaly_model(
        sample_test_data,
        model_path=trained_model,
    )

    assert isinstance(result["feature_comparison"], pd.DataFrame)


def test_evaluate_anomaly_model_does_not_use_target_for_prediction(
    sample_test_data,
    trained_model,
):
    model = joblib.load(trained_model)

    features = prepare_evaluation_features(sample_test_data, model)

    expected_predictions = model.predict(features)

    result = evaluate_anomaly_model(
        sample_test_data,
        model_path=trained_model,
    )

    np.testing.assert_array_equal(
        result["results"]["Prediction"].to_numpy(),
        expected_predictions,
    )


def test_evaluate_anomaly_model_rejects_missing_model(tmp_path):
    model_path = tmp_path / "missing_model.joblib"

    with pytest.raises(ValueError, match="Could not load model"):
        evaluate_anomaly_model(
            pd.DataFrame({
                "Hours_Studied": [10, 20],
                "Attendance": [70, 80],
                "Sleep_Hours": [6, 7],
                "Previous_Scores": [60, 70],
                config.TARGET_COLUMN: [60, 70],
            }),
            model_path=model_path,
        )


def test_evaluate_anomaly_model_loads_default_files(
    sample_train_data,
    sample_test_data,
    tmp_path,
    monkeypatch,
):
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"

    processed_dir.mkdir()
    models_dir.mkdir()

    sample_train_data.to_csv(
        processed_dir / "train.csv",
        index=False,
    )

    sample_test_data.to_csv(
        processed_dir / "test.csv",
        index=False,
    )

    monkeypatch.setattr(
        config,
        "PROCESSED_DATA_DIR",
        processed_dir,
    )

    monkeypatch.setattr(
        config,
        "MODELS_DIR",
        models_dir,
    )

    train_anomaly_model()

    result = evaluate_anomaly_model()

    assert len(result["results"]) == len(sample_test_data)


def test_evaluate_anomaly_model_rejects_missing_test_csv(
    tmp_path,
    monkeypatch,
):
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"

    processed_dir.mkdir()
    models_dir.mkdir()

    monkeypatch.setattr(
        config,
        "PROCESSED_DATA_DIR",
        processed_dir,
    )

    monkeypatch.setattr(
        config,
        "MODELS_DIR",
        models_dir,
    )

    model = IsolationForest(random_state=42)
    model.fit(pd.DataFrame({
        "Hours_Studied": [10, 20, 30],
        "Attendance": [70, 80, 90],
        "Sleep_Hours": [6, 7, 8],
        "Previous_Scores": [60, 70, 80],
    }))

    joblib.dump(
        model,
        models_dir / "anomaly_detection_model.joblib",
    )

    with pytest.raises(ValueError, match="Could not load testing data"):
        evaluate_anomaly_model()


# ---------------------------------------------------------
# print_results tests
# ---------------------------------------------------------

def test_print_results_displays_evaluation_output(
    sample_test_data,
    trained_model,
    capsys,
):
    result = evaluate_anomaly_model(
        sample_test_data,
        model_path=trained_model,
    )

    print_results(result)

    captured = capsys.readouterr()

    assert "Anomaly Detection Evaluation" in captured.out
    assert "Test rows" in captured.out
    assert "Anomalies" in captured.out
    assert "Normal observations" in captured.out
