
"""Load trained models and predict student performance."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# Original input features in the same order used during training.
FEATURE_COLUMNS = (
    list(config.NUMERIC_RANGES.keys())
    + list(config.CATEGORICAL_VALUES.keys())
)


def validate_student_input(student_data):
    """Validate one student's original input data and return a DataFrame."""

    if isinstance(student_data, dict):
        input_df = pd.DataFrame([student_data])

    elif isinstance(student_data, pd.DataFrame):
        input_df = student_data.copy()

    else:
        raise TypeError(
            "Student data must be a dictionary or a pandas DataFrame."
        )

    if input_df.empty:
        raise ValueError("Student input cannot be empty.")

    if len(input_df) != 1:
        raise ValueError("Please provide data for exactly one student.")

    if input_df.columns.duplicated().any():
        raise ValueError("Student input contains duplicate column names.")

    missing_columns = [
        column for column in FEATURE_COLUMNS
        if column not in input_df.columns
    ]

    unexpected_columns = [
        column for column in input_df.columns
        if column not in FEATURE_COLUMNS
    ]

    if missing_columns or unexpected_columns:
        raise ValueError(
            f"Invalid input columns. Missing: {missing_columns}. "
            f"Unexpected: {unexpected_columns}."
        )

    input_df = input_df[FEATURE_COLUMNS].copy()

    # Validate numerical features and their allowed ranges.
    for column, (minimum, maximum) in config.NUMERIC_RANGES.items():

        if (
            not pd.api.types.is_numeric_dtype(input_df[column])
            or pd.api.types.is_bool_dtype(input_df[column])
        ):
            raise ValueError(
                f"{column} must contain numerical values."
            )

        value = input_df[column].iloc[0]

        if not np.isfinite(value):
            raise ValueError(f"{column} must be a finite number.")

        if not minimum <= value <= maximum:
            raise ValueError(
                f"{column} must be between {minimum} and {maximum}."
            )

    # Validate categorical features.
    for column, allowed_values in config.CATEGORICAL_VALUES.items():

        value = input_df[column].iloc[0]

        if not isinstance(value, str):
            raise ValueError(
                f"{column} must contain a valid categorical value."
            )

        if value not in allowed_values:
            raise ValueError(
                f"Invalid value for {column}: {value}. "
                f"Allowed values: {allowed_values}."
            )

    return input_df


class StudentPerformancePredictor:
    """Load the saved models and predict one student's results."""

    def __init__(
        self,
        preprocessor_path=None,
        regression_model_path=None,
        anomaly_model_path=None,
    ):

        models_dir = Path(config.MODELS_DIR)

        self.preprocessor_path = (
            Path(preprocessor_path)
            if preprocessor_path is not None
            else models_dir / "preprocessor.joblib"
        )

        self.regression_model_path = (
            Path(regression_model_path)
            if regression_model_path is not None
            else models_dir / "regression_model.joblib"
        )

        self.anomaly_model_path = (
            Path(anomaly_model_path)
            if anomaly_model_path is not None
            else models_dir / "anomaly_detection_model.joblib"
        )

        self.preprocessor = self._load_model(self.preprocessor_path)
        self.regression_model = self._load_model(
            self.regression_model_path
        )
        self.anomaly_model = self._load_model(self.anomaly_model_path)

        self._validate_model_features()

        logger.info("All prediction models loaded successfully.")

    @staticmethod
    def _load_model(model_path):
        """Load a saved model and report missing files clearly."""

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model file was not found: {model_path}"
            )

        try:
            return joblib.load(model_path)

        except Exception as exc:
            raise RuntimeError(
                f"Could not load model file: {model_path}"
            ) from exc

    def _validate_model_features(self):
        """Check that all models expect the same transformed features."""

        if not hasattr(self.preprocessor, "transform"):
            raise TypeError("The saved preprocessor is invalid.")

        if not hasattr(self.preprocessor, "get_feature_names_out"):
            raise TypeError(
                "The preprocessor does not provide feature names."
            )

        feature_names = list(
            self.preprocessor.get_feature_names_out()
        )

        for model_name, model in (
            ("Regression", self.regression_model),
            ("Anomaly Detection", self.anomaly_model),
        ):

            if not hasattr(model, "predict"):
                raise TypeError(
                    f"{model_name} model does not support prediction."
                )

            if model.n_features_in_ != len(feature_names):
                raise ValueError(
                    f"{model_name} model feature count does not match "
                    "the preprocessor."
                )

            model_feature_names = getattr(
                model, "feature_names_in_", None
            )

            if (
                model_feature_names is not None
                and list(model_feature_names) != feature_names
            ):
                raise ValueError(
                    f"{model_name} model feature names do not match "
                    "the preprocessor."
                )

        if not hasattr(self.anomaly_model, "decision_function"):
            raise TypeError(
                "The anomaly model does not support decision_function."
            )

    def predict(self, student_data):
        """Predict exam score and detect unusual input patterns."""

        input_df = validate_student_input(student_data)

        # Use the saved preprocessor; never fit it on new student data.
        transformed_data = self.preprocessor.transform(input_df)

        feature_names = list(
            self.preprocessor.get_feature_names_out()
        )

        transformed_df = pd.DataFrame(
            transformed_data,
            columns=feature_names,
        )

        if not np.isfinite(transformed_df.to_numpy()).all():
            raise ValueError(
                "Transformed student data contains invalid values."
            )

        # Regression prediction.
        predicted_score = float(
            self.regression_model.predict(transformed_df)[0]
        )

        if not np.isfinite(predicted_score):
            raise ValueError("The predicted exam score is not finite.")

        # Anomaly detection: 1 = Normal, -1 = Anomaly.
        anomaly_prediction = int(
            self.anomaly_model.predict(transformed_df)[0]
        )

        if anomaly_prediction not in (1, -1):
            raise ValueError("The anomaly model returned an invalid result.")

        anomaly_status = (
            "Normal" if anomaly_prediction == 1 else "Anomaly"
        )

        anomaly_score = float(
            self.anomaly_model.decision_function(transformed_df)[0]
        )

        if not np.isfinite(anomaly_score):
            raise ValueError("The anomaly score is not finite.")

        results = {
            "predicted_exam_score": predicted_score,
            "anomaly_prediction": anomaly_prediction,
            "anomaly_status": anomaly_status,
            "anomaly_score": anomaly_score,
        }

        logger.info(
            "Student prediction completed. Status: %s",
            anomaly_status,
        )

        return results


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    print("AI Prediction module is ready.")
    print("Use StudentPerformancePredictor to predict student results.")
