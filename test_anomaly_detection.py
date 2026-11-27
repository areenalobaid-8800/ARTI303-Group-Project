
"""Tests for the Anomaly Detection module."""

import joblib
import numpy as np
import pandas as pd
import pytest

import config
from Anomaly_Detection import (
    prepare_anomaly_features,
    train_anomaly_model,
)


@pytest.fixture
def sample_train_data():
    """Create valid training data with numeric features and target."""

    rng = np.random.default_rng(42)

    return pd.DataFrame({
        "Hours_Studied": rng.normal(20, 5, 100),
        "Attendance": rng.normal(80, 10, 100),
        "Sleep_Hours": rng.normal(7, 1, 100),
        "Previous_Scores": rng.normal(70, 10, 100),
        config.TARGET_COLUMN: rng.normal(67, 4, 100),
    })


def test_prepare_anomaly_features_returns_dataframe(sample_train_data):
    result = prepare_anomaly_features(sample_train_data)

    assert isinstance(result, pd.DataFrame)


def test_prepare_anomaly_features_excludes_target(sample_train_data):
    result = prepare_anomaly_features(sample_train_data)

    assert config.TARGET_COLUMN not in result.columns


def test_prepare_anomaly_features_preserves_rows(sample_train_data):
    result = prepare_anomaly_features(sample_train_data)

    assert len(result) == len(sample_train_data)


def test_prepare_anomaly_features_preserves_feature_names(sample_train_data):
    result = prepare_anomaly_features(sample_train_data)

    expected = [
        col for col in sample_train_data.columns
        if col != config.TARGET_COLUMN
    ]

    assert list(result.columns) == expected


def test_prepare_anomaly_features_rejects_non_dataframe():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        prepare_anomaly_features([1, 2, 3])


def test_prepare_anomaly_features_rejects_empty_dataframe():
    empty_df = pd.DataFrame(
        columns=["Hours_Studied", config.TARGET_COLUMN]
    )

    with pytest.raises(ValueError, match="empty"):
        prepare_anomaly_features(empty_df)


def test_prepare_anomaly_features_rejects_missing_target():
    df = pd.DataFrame({
        "Hours_Studied": [10, 20, 30]
    })

    with pytest.raises(KeyError, match="Missing target"):
        prepare_anomaly_features(df)


def test_prepare_anomaly_features_rejects_duplicate_columns():
    df = pd.DataFrame(
        [[10, 20, 60], [15, 25, 70]],
        columns=[
            "Hours_Studied",
            "Hours_Studied",
            config.TARGET_COLUMN,
        ],
    )

    with pytest.raises(ValueError, match="duplicate"):
        prepare_anomaly_features(df)


def test_prepare_anomaly_features_rejects_missing_values(sample_train_data):
    df = sample_train_data.copy()
    df.loc[0, "Hours_Studied"] = np.nan

    with pytest.raises(ValueError, match="Missing values"):
        prepare_anomaly_features(df)


def test_prepare_anomaly_features_rejects_nonnumeric_features(sample_train_data):
    df = sample_train_data.copy()
    df["Hours_Studied"] = df["Hours_Studied"].astype(str)

    with pytest.raises(TypeError, match="must contain numeric"):
        prepare_anomaly_features(df)


@pytest.mark.parametrize("invalid_value", [np.inf, -np.inf])
def test_prepare_anomaly_features_rejects_infinite_values(
    sample_train_data,
    invalid_value,
):
    df = sample_train_data.copy()
    df.loc[0, "Hours_Studied"] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        prepare_anomaly_features(df)


def test_prepare_anomaly_features_rejects_no_features():
    df = pd.DataFrame({
        config.TARGET_COLUMN: [60, 70, 80]
    })

    with pytest.raises(ValueError, match="No features"):
        prepare_anomaly_features(df)


def test_train_anomaly_model_returns_expected_keys(
    sample_train_data,
    tmp_path,
):
    result = train_anomaly_model(
        sample_train_data,
        model_path=tmp_path / "model.joblib",
    )

    expected_keys = {
        "model",
        "predictions",
        "anomaly_scores",
        "anomaly_count",
        "normal_count",
        "contamination",
        "model_path",
    }

    assert set(result.keys()) == expected_keys


def test_train_anomaly_model_uses_isolation_forest(
    sample_train_data,
    tmp_path,
):
    from sklearn.ensemble import IsolationForest

    result = train_anomaly_model(
        sample_train_data,
        model_path=tmp_path / "model.joblib",
    )

    assert isinstance(result["model"], IsolationForest)


def test_train_anomaly_model_excludes_target_from_model(
    sample_train_data,
    tmp_path,
):
    result = train_anomaly_model(
        sample_train_data,
        model_path=tmp_path / "model.joblib",
    )

    assert config.TARGET_COLUMN not in list(
        result["model"].feature_names_in_
    )


def test_train_anomaly_model_uses_all_expected_features(
    sample_train_data,
    tmp_path,
):
    result = train_anomaly_model(
        sample_train_data,
        model_path=tmp_path / "model.joblib",
    )

    expected = [
        col for col in sample_train_data.columns
        if col != config.TARGET_COLUMN
    ]

    assert list(result["model"].feature_names_in_) == expected


def test_train_anomaly_model_returns_valid_predictions(
    sample_train_data,
    tmp_path,
):
    result = train_anomaly_model(
        sample_train_data,
        model_path=tmp_path / "model.joblib",
    )

    assert set(np.unique(result["predictions"])).issubset({-1, 1})


def test_train_anomaly_model_returns_finite_scores(
    sample_train_data,
    tmp_path,
):
    result = train_anomaly_model(
        sample_train_data,
        model_path=tmp_path / "model.joblib",
    )

    assert np.isfinite(result["anomaly_scores"]).all()


def test_train_anomaly_model_counts_are_consistent(
    sample_train_data,
    tmp_path,
):
    result = train_anomaly_model(
        sample_train_data,
        model_path=tmp_path / "model.joblib",
    )

    assert result["anomaly_count"] == (
        result["predictions"] == -1
    ).sum()

    assert result["normal_count"] == (
        result["predictions"] == 1
    ).sum()

    assert (
        result["anomaly_count"] + result["normal_count"]
        == len(sample_train_data)
    )


def test_train_anomaly_model_uses_requested_contamination(
    sample_train_data,
    tmp_path,
):
    result = train_anomaly_model(
        sample_train_data,
        model_path=tmp_path / "model.joblib",
        contamination=0.1,
    )

    assert result["contamination"] == 0.1
    assert result["model"].contamination == 0.1


@pytest.mark.parametrize(
    "invalid_contamination",
    [0, -0.1, 0.6, 1, "0.05", None],
)
def test_train_anomaly_model_rejects_invalid_contamination(
    sample_train_data,
    tmp_path,
    invalid_contamination,
):
    with pytest.raises((TypeError, ValueError)):
        train_anomaly_model(
            sample_train_data,
            model_path=tmp_path / "model.joblib",
            contamination=invalid_contamination,
        )


def test_train_anomaly_model_saves_file(
    sample_train_data,
    tmp_path,
):
    model_path = tmp_path / "models" / "model.joblib"

    train_anomaly_model(
        sample_train_data,
        model_path=model_path,
    )

    assert model_path.exists()


def test_train_anomaly_model_can_load_saved_model(
    sample_train_data,
    tmp_path,
):
    model_path = tmp_path / "model.joblib"

    train_anomaly_model(
        sample_train_data,
        model_path=model_path,
    )

    loaded_model = joblib.load(model_path)

    assert hasattr(loaded_model, "predict")
    assert hasattr(loaded_model, "decision_function")


def test_train_anomaly_model_predictions_are_reproducible(
    sample_train_data,
    tmp_path,
):
    first = train_anomaly_model(
        sample_train_data,
        model_path=tmp_path / "first.joblib",
    )

    second = train_anomaly_model(
        sample_train_data,
        model_path=tmp_path / "second.joblib",
    )

    np.testing.assert_array_equal(
        first["predictions"],
        second["predictions"],
    )

    np.testing.assert_allclose(
        first["anomaly_scores"],
        second["anomaly_scores"],
    )


def test_train_anomaly_model_loads_default_training_data(
    sample_train_data,
    tmp_path,
    monkeypatch,
):
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"

    processed_dir.mkdir()

    sample_train_data.to_csv(
        processed_dir / "train.csv",
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

    result = train_anomaly_model()

    assert result["model_path"].exists()
    assert len(result["predictions"]) == len(sample_train_data)


def test_train_anomaly_model_rejects_missing_default_csv(
    tmp_path,
    monkeypatch,
):
    processed_dir = tmp_path / "missing"

    monkeypatch.setattr(
        config,
        "PROCESSED_DATA_DIR",
        processed_dir,
    )

    with pytest.raises(ValueError, match="Could not load training data"):
        train_anomaly_model()


def test_train_anomaly_model_rejects_invalid_dataframe(
    tmp_path,
):
    with pytest.raises(TypeError, match="pandas DataFrame"):
        train_anomaly_model(
            [1, 2, 3],
            model_path=tmp_path / "model.joblib",
        )
