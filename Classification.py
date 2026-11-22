
import logging
import joblib
import pandas as pd

from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from config import (
    PROJECT_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE
)


# ==================================================
# 1. Configuration
# ==================================================

TARGET_COLUMN = "Exam_Score"
CLASS_COLUMN = "Performance_Level"

LOW_LIMIT = 60
HIGH_LIMIT = 80

CLASS_LABELS = ["Low", "Medium", "High"]

VALIDATION_SIZE = 0.20

MODEL_DIR = PROJECT_DIR / "models"


# ==================================================
# 2. Logging Configuration
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ==================================================
# 3. Load Training Data
# ==================================================

def load_data():
    """Load and validate the transformed training dataset."""

    file_path = PROCESSED_DATA_DIR / "train_transformed.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Training data file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("Training dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Required column '{TARGET_COLUMN}' is missing."
        )

    if df.columns.duplicated().any():
        raise ValueError(
            "Dataset contains duplicate column names."
        )

    logger.info("Training data loaded successfully.")
    logger.info("Dataset shape: %s", df.shape)

    return df


# ==================================================
# 4. Create Performance Levels
# ==================================================

def create_performance_level(df):
    """Create Low, Medium, and High classes from Exam_Score."""

    df = df.copy()

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Exam_Score contains missing or invalid values."
        )

    if not df[TARGET_COLUMN].between(0, 100).all():
        raise ValueError(
            "Exam_Score values must be between 0 and 100."
        )

    df[CLASS_COLUMN] = pd.cut(
        df[TARGET_COLUMN],
        bins=[
            float("-inf"),
            LOW_LIMIT,
            HIGH_LIMIT,
            float("inf")
        ],
        labels=CLASS_LABELS,
        right=False
    )

    if df[CLASS_COLUMN].isna().any():
        raise ValueError(
            "Some records could not be assigned a performance level."
        )

    distribution = df[CLASS_COLUMN].value_counts(
        sort=False
    )

    print("\nPerformance Level Distribution:")
    print(distribution)

    if (distribution == 0).any():
        raise ValueError(
            "One or more performance classes have no samples."
        )

    logger.info("Performance levels created successfully.")

    return df


# ==================================================
# 5. Prepare Features and Target
# ==================================================

def prepare_features(df):
    """Separate input features from the classification target."""

    # Exam_Score is used only to create Performance_Level.
    # It is excluded from X to prevent target leakage.

    X = df.drop(
        columns=[TARGET_COLUMN, CLASS_COLUMN]
    )

    y = df[CLASS_COLUMN].astype(str)

    if X.empty:
        raise ValueError("No input features are available.")

    if X.isna().any().any():
        raise ValueError(
            "Input features contain missing values."
        )

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in X.dtypes
    ):
        raise TypeError(
            "All input features must be numeric."
        )

    if y.nunique() != len(CLASS_LABELS):
        raise ValueError(
            "All three performance classes must be present."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "Each class must contain at least two samples."
        )

    feature_names = X.columns.tolist()

    print("\nNumber of input features:", X.shape[1])
    print("Target column:", CLASS_COLUMN)
    print("Exam_Score excluded from input features.")

    logger.info("Features and target prepared successfully.")

    return X, y, feature_names


# ==================================================
# 6. Create Model Configurations
# ==================================================

def get_model_candidates():
    """Return model configurations and balancing strategies."""

    candidates = []

    balancing_strategies = [
        "none",
        "class_weight",
        "random_over_sampling"
    ]

    decision_tree_configs = [
        {
            "max_depth": None,
            "min_samples_leaf": 1
        },
        {
            "max_depth": 10,
            "min_samples_leaf": 2
        },
        {
            "max_depth": 5,
            "min_samples_leaf": 5
        }
    ]

    random_forest_configs = [
        {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt"
        },
        {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_leaf": 2,
            "max_features": "sqrt"
        },
        {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_leaf": 3,
            "max_features": "sqrt"
        }
    ]

    for config in decision_tree_configs:
        for strategy in balancing_strategies:
            candidates.append({
                "model_name": "Decision Tree",
                "model_type": "Decision_Tree",
                "parameters": config,
                "strategy": strategy
            })

    for config in random_forest_configs:
        for strategy in balancing_strategies:
            candidates.append({
                "model_name": "Random Forest",
                "model_type": "Random_Forest_Classifier",
                "parameters": config,
                "strategy": strategy
            })

    return candidates


# ==================================================
# 7. Build and Train a Model
# ==================================================

def build_model(model_type, parameters, strategy):
    """Create a classifier using the selected configuration."""

    model_parameters = parameters.copy()

    if strategy == "class_weight":
        model_parameters["class_weight"] = "balanced"

    if model_type == "Decision_Tree":
        return DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            **model_parameters
        )

    if model_type == "Random_Forest_Classifier":
        return RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **model_parameters
        )

    raise ValueError(f"Unknown model type: {model_type}")


def fit_with_strategy(model, X_train, y_train, strategy):
    """Apply the selected balancing strategy to training data only."""

    if strategy == "random_over_sampling":

        sampler = RandomOverSampler(
            random_state=RANDOM_STATE
        )

        X_resampled, y_resampled = sampler.fit_resample(
            X_train,
            y_train
        )

        model.fit(
            X_resampled,
            y_resampled
        )

    else:
        model.fit(
            X_train,
            y_train
        )

    return model


# ==================================================
# 8. Evaluate a Model
# ==================================================

def evaluate_model(model, X_val, y_val, model_name):
    """Evaluate a trained classifier on untouched validation data."""

    predictions = model.predict(X_val)

    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_val,
            predictions
        ),
        "Macro Precision": precision_score(
            y_val,
            predictions,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0
        ),
        "Macro Recall": recall_score(
            y_val,
            predictions,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0
        ),
        "Macro F1-Score": f1_score(
            y_val,
            predictions,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0
        )
    }

    return results


# ==================================================
# 9. Compare and Select the Best Configurations
# ==================================================

def train_and_compare_models(X, y):
    """Compare configurations and select the best one per model type."""

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nOriginal Training Distribution:")
    print(y_train.value_counts())

    print("\nValidation Distribution:")
    print(y_val.value_counts())

    candidates = get_model_candidates()

    all_results = []
    best_models = {}

    for candidate in candidates:

        model_name = candidate["model_name"]
        model_type = candidate["model_type"]
        parameters = candidate["parameters"]
        strategy = candidate["strategy"]

        model = build_model(
            model_type,
            parameters,
            strategy
        )

        model = fit_with_strategy(
            model,
            X_train,
            y_train,
            strategy
        )

        metrics = evaluate_model(
            model,
            X_val,
            y_val,
            model_name
        )

        result = {
            **metrics,
            "Strategy": strategy,
            "Parameters": str(parameters),
            "Model Type": model_type
        }

        all_results.append(result)

        current_best = best_models.get(model_type)

        # Select by Macro F1, then Balanced Accuracy.

        current_score = (
            metrics["Macro F1-Score"],
            metrics["Balanced Accuracy"]
        )

        if current_best is None or current_score > current_best["score"]:

            best_models[model_type] = {
                "model": model,
                "score": current_score,
                "strategy": strategy,
                "parameters": parameters,
                "metrics": metrics
            }

    comparison_df = pd.DataFrame(all_results)

    comparison_df = comparison_df.sort_values(
        by=["Macro F1-Score", "Balanced Accuracy"],
        ascending=False
    )

    print("\n========== ALL MODEL CONFIGURATIONS ==========")
    print(comparison_df.to_string(index=False))

    print("\n========== SELECTED MODELS ==========")

    for model_type, selected in best_models.items():

        print(f"\n{model_type}")
        print("Selected strategy:", selected["strategy"])
        print("Selected parameters:", selected["parameters"])
        print("Macro F1:", selected["metrics"]["Macro F1-Score"])
        print(
            "Balanced Accuracy:",
            selected["metrics"]["Balanced Accuracy"]
        )

    return best_models, comparison_df


# ==================================================
# 10. Train Final Models on All Training Data
# ==================================================

def train_final_models(X, y, best_models):
    """Retrain selected configurations using all available training data."""

    final_models = {}

    for model_type, selected in best_models.items():

        strategy = selected["strategy"]

        model = build_model(
            model_type,
            selected["parameters"],
            strategy
        )

        model = fit_with_strategy(
            model,
            X,
            y,
            strategy
        )

        final_models[model_type] = model

        logger.info(
            "Final %s model trained successfully.",
            model_type
        )

    return final_models


# ==================================================
# 11. Save Models and Metadata
# ==================================================

def save_models(final_models, best_models, feature_names):
    """Save final classifiers, metadata, and model comparison results."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for model_name, model in final_models.items():

        file_path = MODEL_DIR / f"{model_name}.pkl"

        joblib.dump(
            model,
            file_path
        )

        print(f"\n{model_name} saved successfully.")
        print(f"Path: {file_path}")

    metadata = {
        "feature_names": feature_names,
        "target_column": CLASS_COLUMN,
        "source_target_column": TARGET_COLUMN,
        "class_labels": CLASS_LABELS,
        "low_limit": LOW_LIMIT,
        "high_limit": HIGH_LIMIT,
        "selected_models": {
            model_type: {
                "strategy": selected["strategy"],
                "parameters": selected["parameters"],
                "validation_metrics": selected["metrics"]
            }
            for model_type, selected in best_models.items()
        }
    }

    metadata_path = MODEL_DIR / "Classification_Metadata.pkl"

    joblib.dump(
        metadata,
        metadata_path
    )

    print("\nClassification metadata saved successfully.")
    print(f"Path: {metadata_path}")

    logger.info("All classification files saved successfully.")


# ==================================================
# 12. Main
# ==================================================

def main():
    """Execute the complete classification workflow."""

    try:

        df = load_data()

        df = create_performance_level(df)

        X, y, feature_names = prepare_features(df)

        best_models, comparison_df = train_and_compare_models(
            X,
            y
        )

        final_models = train_final_models(
            X,
            y,
            best_models
        )

        save_models(
            final_models,
            best_models,
            feature_names
        )

        print(
            "\nClassification model development "
            "completed successfully!"
        )

    except (FileNotFoundError, ValueError, TypeError):

        logger.exception(
            "Data or configuration error occurred."
        )
        raise

    except Exception:

        logger.exception(
            "Unexpected error occurred."
        )
        raise


if __name__ == "__main__":
    main()
