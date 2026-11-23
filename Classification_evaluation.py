
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from config import (
    PROJECT_DIR,
    PROCESSED_DATA_DIR
)


# ==================================================
# 1. Configuration
# ==================================================

TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_transformed.csv"

MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
VISUALIZATIONS_DIR = PROJECT_DIR / "visualizations"

TARGET_COLUMN = "Exam_Score"

CLASS_LABELS = ["Low", "Medium", "High"]

LOW_LIMIT = 60
HIGH_LIMIT = 80


# ==================================================
# 2. Logging Configuration
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


# ==================================================
# 3. Create Performance Classes
# ==================================================

def create_performance_labels(scores):
    """Convert exam scores into Low, Medium, and High classes."""

    scores = pd.to_numeric(
        pd.Series(scores),
        errors="coerce"
    )

    if scores.isna().any():
        raise ValueError(
            "Actual exam scores contain missing or invalid values."
        )

    if not scores.between(0, 100).all():
        raise ValueError(
            "Exam scores must be between 0 and 100."
        )

    labels = pd.cut(
        scores,
        bins=[
            float("-inf"),
            LOW_LIMIT,
            HIGH_LIMIT,
            float("inf")
        ],
        labels=CLASS_LABELS,
        right=False
    )

    return labels.astype(str).to_numpy()


# ==================================================
# 4. Validate Actual and Predicted Labels
# ==================================================

def validate_actual_labels(y_actual):
    """Validate the actual classification labels."""

    y_actual = np.asarray(y_actual)

    if len(y_actual) == 0:
        raise ValueError("Actual labels cannot be empty.")

    if pd.isna(y_actual).any():
        raise ValueError("Actual labels contain missing values.")

    if not np.isin(y_actual, CLASS_LABELS).all():
        raise ValueError("Actual labels contain unknown classes.")


def validate_predictions(y_actual, y_pred, model_name):
    """Validate model predictions."""

    y_actual = np.asarray(y_actual)
    y_pred = np.asarray(y_pred)

    if len(y_pred) != len(y_actual):
        raise ValueError(
            f"{model_name}: Number of predictions "
            "does not match actual labels."
        )

    if len(y_pred) == 0:
        raise ValueError("Predictions cannot be empty.")

    if pd.isna(y_pred).any():
        raise ValueError(
            f"{model_name}: Predictions contain missing values."
        )

    if not np.isin(y_pred, CLASS_LABELS).all():
        raise ValueError(
            f"{model_name}: Predictions contain unknown classes."
        )


# ==================================================
# 5. Calculate Classification Metrics
# ==================================================

def calculate_classification_metrics(y_actual, y_pred):
    """Calculate classification evaluation metrics."""

    validate_actual_labels(y_actual)
    validate_predictions(y_actual, y_pred, "Classification Model")

    return {
        "Accuracy": accuracy_score(y_actual, y_pred),

        "Balanced Accuracy": balanced_accuracy_score(
            y_actual,
            y_pred
        ),

        "Precision": precision_score(
            y_actual,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0
        ),

        "Recall": recall_score(
            y_actual,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0
        ),

        "F1-Score": f1_score(
            y_actual,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0
        )
    }


# ==================================================
# 6. Analyze Classification Errors
# ==================================================

def analyze_classification_errors(y_actual, y_pred):
    """Identify incorrectly classified performance levels."""

    errors = pd.DataFrame({
        "Actual": np.asarray(y_actual),
        "Predicted": np.asarray(y_pred)
    })

    errors = errors[
        errors["Actual"] != errors["Predicted"]
    ].copy()

    if errors.empty:
        return pd.DataFrame(
            columns=["Actual", "Predicted", "Count"]
        )

    error_summary = (
        errors.groupby(
            ["Actual", "Predicted"],
            observed=False
        )
        .size()
        .reset_index(name="Count")
    )

    return error_summary


# ==================================================
# 7. Evaluate Classification Models
# ==================================================

def evaluate_classification_models():
    """Evaluate saved classification models on test data."""

    # Load test data
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test data not found: {TEST_DATA_PATH}"
        )

    test_df = pd.read_csv(TEST_DATA_PATH)

    if test_df.empty:
        raise ValueError("Test dataset is empty.")

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(
            f"{TARGET_COLUMN} column is missing from test data."
        )

    # Create actual performance labels
    y_actual = create_performance_labels(
        test_df[TARGET_COLUMN]
    )

    # Load model metadata
    metadata_path = MODELS_DIR / "Classification_Metadata.pkl"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Classification metadata not found: {metadata_path}"
        )

    metadata = joblib.load(metadata_path)

    feature_names = metadata["feature_names"]

    if metadata.get("class_labels") != CLASS_LABELS:
        raise ValueError(
            "Class labels in metadata do not match evaluation labels."
        )

    # Check required features
    missing_features = [
        feature for feature in feature_names
        if feature not in test_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing test features: {missing_features}"
        )

    # Use the same feature order as training
    X_test = test_df[feature_names]

    if X_test.isna().any().any():
        raise ValueError(
            "Test features contain missing values."
        )

    # Saved models
    models = {
        "Decision Tree": MODELS_DIR / "Decision_Tree.pkl",
        "Random Forest": MODELS_DIR / "Random_Forest_Classifier.pkl"
    }

    results = []
    predictions_data = pd.DataFrame({
        "Actual_Performance": y_actual
    })

    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Evaluate each model
    for model_name, model_path in models.items():

        logging.info("Evaluating %s...", model_name)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        model = joblib.load(model_path)

        # Generate predictions
        y_pred = np.asarray(model.predict(X_test))

        validate_predictions(
            y_actual,
            y_pred,
            model_name
        )

        # Calculate metrics
        metrics = calculate_classification_metrics(
            y_actual,
            y_pred
        )

        results.append({
            "Model": model_name,
            **metrics
        })

        predictions_data[f"{model_name}_Prediction"] = y_pred

        # Create confusion matrix
        cm = confusion_matrix(
            y_actual,
            y_pred,
            labels=CLASS_LABELS
        )

        cm_df = pd.DataFrame(
            cm,
            index=CLASS_LABELS,
            columns=CLASS_LABELS
        )

        cm_df.to_csv(
            RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.csv"
        )

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(7, 5))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=CLASS_LABELS
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            values_format="d"
        )

        ax.set_title(f"{model_name} Confusion Matrix")

        fig.tight_layout()

        fig.savefig(
            VISUALIZATIONS_DIR /
            f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        # Analyze errors
        error_summary = analyze_classification_errors(
            y_actual,
            y_pred
        )

        error_summary.to_csv(
            RESULTS_DIR /
            f"{model_name.lower().replace(' ', '_')}_errors.csv",
            index=False
        )

        print(f"\n{model_name} Classification Errors:")
        print(error_summary.to_string(index=False))

    # Save evaluation results
    results_df = pd.DataFrame(results)

    print("\nClassification Model Evaluation Results")
    print(results_df.to_string(index=False))

    results_df.to_csv(
        RESULTS_DIR / "classification_evaluation_results.csv",
        index=False
    )

    predictions_data.to_csv(
        RESULTS_DIR / "classification_predictions.csv",
        index=False
    )

    logging.info("Classification evaluation completed successfully.")

    return results_df


# ==================================================
# 8. Main
# ==================================================

if __name__ == "__main__":
    evaluate_classification_models()
