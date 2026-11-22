import logging

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config import PROJECT_DIR, PROCESSED_DATA_DIR, TARGET_COLUMN


# Set project paths
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_transformed.csv"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


def validate_actual_scores(y_test):
    """Check that actual exam scores contain valid numeric values."""

    if not np.isfinite(y_test).all():
        raise ValueError(
            "Actual scores contain invalid numeric values."
        )


def validate_predictions(y_test, y_pred, model_name):
    """Validate the model predictions."""

    if len(y_pred) != len(y_test):
        raise ValueError(
            f"{model_name}: Number of predictions does not "
            "match the number of actual scores."
        )

    if not np.isfinite(y_pred).all():
        raise ValueError(
            f"{model_name}: Predictions contain invalid "
            "numeric values."
        )


def calculate_regression_metrics(y_test, y_pred):
    """Calculate regression evaluation metrics."""

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


def evaluate_regression_models():
    """Evaluate saved regression models using the test dataset."""

    try:
        # Load test data
        test_df = pd.read_csv(TEST_DATA_PATH)

        # Check target column
        if TARGET_COLUMN not in test_df.columns:
            raise ValueError(
                f"{TARGET_COLUMN} column is missing from test data."
            )

        # Separate features and target
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        y_test = test_df[TARGET_COLUMN]

        # Validate actual scores
        validate_actual_scores(y_test)

        logging.info("Test data loaded successfully.")
        logging.info("Number of test records: %d", len(test_df))

        # Define saved models
        models = {
            "Linear Regression": MODELS_DIR / "linear_regression.joblib",
            "Random Forest": MODELS_DIR / "random_forest_regressor.joblib"
        }

        results = []
        predictions_data = pd.DataFrame({
            "Actual_Exam_Score": y_test
        })
        all_predictions = {}

        # Evaluate each model
        for model_name, model_path in models.items():

            logging.info("Evaluating %s...", model_name)

            # Load trained model
            model = joblib.load(model_path)

            # Generate predictions
            y_pred = np.asarray(model.predict(X_test))

            # Validate predictions
            validate_predictions(y_test, y_pred, model_name)

            logging.info(
                "%s: Prediction validation passed.",
                model_name
            )

            # Save predictions
            predictions_data[model_name] = y_pred
            all_predictions[model_name] = y_pred

            # Calculate evaluation metrics
            metrics = calculate_regression_metrics(y_test, y_pred)

            results.append({
                "Model": model_name,
                **metrics
            })

        # Display comparison table
        results_df = pd.DataFrame(results)

        print("\nRegression Model Evaluation Results")
        print(results_df.to_string(index=False))

        # Create results directory if it does not exist
        RESULTS_DIR.mkdir(exist_ok=True)

        # Save evaluation metrics
        results_df.to_csv(
            RESULTS_DIR / "regression_evaluation_results.csv",
            index=False
        )

        # Save actual and predicted scores
        predictions_data.to_csv(
            RESULTS_DIR / "regression_predictions.csv",
            index=False
        )

        # Create Actual vs Predicted plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, (model_name, y_pred) in zip(
            axes, all_predictions.items()
        ):
            ax.scatter(y_test, y_pred, alpha=0.5)

            min_value = min(y_test.min(), y_pred.min())
            max_value = max(y_test.max(), y_pred.max())

            ax.plot(
                [min_value, max_value],
                [min_value, max_value],
                linestyle="--"
            )

            ax.set_title(model_name)
            ax.set_xlabel("Actual Exam Scores")
            ax.set_ylabel("Predicted Exam Scores")
            ax.grid(True)

        fig.suptitle("Actual vs Predicted Exam Scores")
        fig.tight_layout()

        fig.savefig(
            RESULTS_DIR / "actual_vs_predicted.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        logging.info(
            "Evaluation files and plot saved successfully."
        )
        logging.info("Results saved in: %s", RESULTS_DIR)

        return results_df

    except FileNotFoundError as error:
        logging.error("Required file not found: %s", error)

    except Exception:
        logging.exception(
            "An error occurred during model evaluation."
        )


if __name__ == "__main__":
    evaluate_regression_models()