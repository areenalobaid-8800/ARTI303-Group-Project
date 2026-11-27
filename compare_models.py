
"""
Compare Regression and Anomaly Detection algorithms.

Regression:
- Random Forest
- Decision Tree
- Gradient Boosting
- Dummy Baseline

Anomaly Detection:
- Isolation Forest
- Local Outlier Factor
- One-Class SVM

Regression model selection:
- Uses 5-Fold Cross-Validation on training data.
- Selects the model with the lowest Cross-Validation MAE.
- Evaluates the selected model on the held-out test data.

Anomaly Detection:
- Excludes Exam_Score from all detection features.
- Compares anomaly counts, percentages, and execution times.
- Uses Jaccard agreement with Isolation Forest as a reference.
- Jaccard agreement is NOT detection accuracy because no ground-truth
  anomaly labels are available.
"""

import logging
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    IsolationForest,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import config


logger = logging.getLogger(__name__)


# --------------------------------------------------
# 1. Load and validate processed data
# --------------------------------------------------

def load_data():
    """Load and validate the processed training and testing datasets."""

    train_path = config.PROCESSED_DATA_DIR / "train.csv"
    test_path = config.PROCESSED_DATA_DIR / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_path}. "
            "Run the data pipeline first."
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Testing file not found: {test_path}. "
            "Run the data pipeline first."
        )

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

    except (OSError, pd.errors.ParserError) as exc:
        logger.exception("Failed to load processed datasets")
        raise ValueError(
            "Could not read the processed training or testing data."
        ) from exc

    target = config.TARGET_COLUMN

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Training or testing dataset is empty."
        )

    if train_df.columns.duplicated().any():
        raise ValueError(
            "Training data contains duplicate columns."
        )

    if test_df.columns.duplicated().any():
        raise ValueError(
            "Testing data contains duplicate columns."
        )

    if target not in train_df.columns:
        raise KeyError(
            f"Target column '{target}' missing in training data."
        )

    if target not in test_df.columns:
        raise KeyError(
            f"Target column '{target}' missing in testing data."
        )

    X_train = train_df.drop(columns=[target])
    X_test = test_df.drop(columns=[target])

    y_train = train_df[target]
    y_test = test_df[target]

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and testing feature columns do not match."
        )

    if X_train.empty or X_test.empty:
        raise ValueError(
            "Training or testing data contains no input features."
        )

    if len(X_train) < 5:
        raise ValueError(
            "At least 5 training rows are required for 5-Fold Cross-Validation."
        )

    if y_train.isna().any() or y_test.isna().any():
        raise ValueError(
            "Target column contains missing values."
        )

    for name, features in [
        ("Training", X_train),
        ("Testing", X_test),
    ]:

        if features.isna().any().any():
            raise ValueError(
                f"{name} features contain missing values."
            )

        for col in features.columns:
            if not pd.api.types.is_numeric_dtype(features[col]):
                raise TypeError(
                    f"{name} feature '{col}' must be numeric."
                )

        values = features.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"{name} features contain non-finite values."
            )

    for name, target_values in [
        ("Training", y_train),
        ("Testing", y_test),
    ]:

        if not pd.api.types.is_numeric_dtype(target_values):
            raise TypeError(
                f"{name} target must be numeric."
            )

        if not np.isfinite(
            target_values.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{name} target contains non-finite values."
            )

    logger.info(
        "Processed datasets loaded: %d training rows, %d testing rows, %d features",
        len(X_train),
        len(X_test),
        X_train.shape[1],
    )

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# 2. Compare Regression algorithms
# --------------------------------------------------

def compare_regression(X_train, X_test, y_train, y_test):
    """
    Compare Regression algorithms using 5-Fold Cross-Validation.

    Model selection is based on the lowest Cross-Validation MAE.
    The held-out test set is used only for final evaluation.
    """

    if len(X_train) < 5:
        raise ValueError(
            "At least 5 training rows are required for 5-Fold Cross-Validation."
        )

    if not (
        len(X_train) == len(y_train)
        and len(X_test) == len(y_test)
    ):
        raise ValueError(
            "Feature and target row counts do not match."
        )

    models = {

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),

        "Decision Tree": DecisionTreeRegressor(
            random_state=config.RANDOM_STATE,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=config.RANDOM_STATE,
        ),

        "Dummy Baseline": DummyRegressor(
            strategy="mean",
        ),
    }

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    results = []

    for name, model in models.items():

        logger.info(
            "Starting Cross-Validation for %s",
            name,
        )

        cv_start = time.perf_counter()

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )

        cv_time = time.perf_counter() - cv_start

        cv_mae = -cv_scores

        mean_cv_mae = float(np.mean(cv_mae))
        std_cv_mae = float(np.std(cv_mae))

        # Fit the model using the complete training dataset.
        training_start = time.perf_counter()

        model.fit(X_train, y_train)

        training_time = time.perf_counter() - training_start

        # Evaluate on the held-out test dataset.
        testing_start = time.perf_counter()

        predictions = model.predict(X_test)

        testing_time = time.perf_counter() - testing_start

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        r2 = r2_score(
            y_test,
            predictions,
        )

        results.append({

            "Algorithm": name,

            "Cross_Validation_MAE": mean_cv_mae,

            "Cross_Validation_MAE_STD": std_cv_mae,

            "Test_MAE": mae,

            "Test_RMSE": rmse,

            "Test_R2": r2,

            "Cross_Validation_Time_Seconds": cv_time,

            "Training_Time_Seconds": training_time,

            "Testing_Time_Seconds": testing_time,
        })

        logger.info(
            "%s completed: CV MAE = %.4f, Test MAE = %.4f",
            name,
            mean_cv_mae,
            mae,
        )

    results_df = pd.DataFrame(results)

    # Select the best model using Cross-Validation MAE only.
    results_df = results_df.sort_values(
        by="Cross_Validation_MAE",
        ascending=True,
    ).reset_index(drop=True)

    results_df["CV_Rank"] = (
        np.arange(1, len(results_df) + 1)
    )

    return results_df


# --------------------------------------------------
# 3. Compare Anomaly Detection algorithms
# --------------------------------------------------

def compare_anomaly_detection(X_train, X_test):
    """
    Compare anomaly detection algorithms.

    Exam_Score must not be included in X_train or X_test.

    No accuracy-based metrics are calculated because the dataset
    does not contain ground-truth anomaly labels.
    """

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(
            "X_train must be a pandas DataFrame."
        )

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            "X_test must be a pandas DataFrame."
        )

    if X_train.empty or X_test.empty:
        raise ValueError(
            "Training or testing features are empty."
        )

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and testing feature columns do not match."
        )

    target = config.TARGET_COLUMN

    if target in X_train.columns or target in X_test.columns:
        raise ValueError(
            f"Target column '{target}' must be excluded from anomaly detection."
        )

    if len(X_train) < 2:
        raise ValueError(
            "At least 2 training rows are required."
        )

    if len(X_test) < 1:
        raise ValueError(
            "At least 1 testing row is required."
        )

    for name, features in [
        ("Training", X_train),
        ("Testing", X_test),
    ]:

        if features.isna().any().any():
            raise ValueError(
                f"{name} features contain missing values."
            )

        for col in features.columns:
            if not pd.api.types.is_numeric_dtype(features[col]):
                raise TypeError(
                    f"{name} feature '{col}' must be numeric."
                )

        if not np.isfinite(
            features.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{name} features contain non-finite values."
            )

    contamination = 0.05

    models = {

        "Isolation Forest": IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),

        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=min(20, len(X_train) - 1),
            contamination=contamination,
            novelty=True,
        ),

        "One-Class SVM": OneClassSVM(
            nu=contamination,
            kernel="rbf",
            gamma="scale",
        ),
    }

    predictions = {}

    results = []

    for name, model in models.items():

        logger.info(
            "Starting Anomaly Detection comparison for %s",
            name,
        )

        # Measure training time separately.
        training_start = time.perf_counter()

        model.fit(X_train)

        training_time = time.perf_counter() - training_start

        # Measure testing time separately.
        testing_start = time.perf_counter()

        labels = model.predict(X_test)

        testing_time = time.perf_counter() - testing_start

        predictions[name] = labels

        anomaly_count = int(
            (labels == -1).sum()
        )

        normal_count = int(
            (labels == 1).sum()
        )

        anomaly_percentage = (
            anomaly_count / len(labels)
        ) * 100

        results.append({

            "Algorithm": name,

            "Normal_Count": normal_count,

            "Anomaly_Count": anomaly_count,

            "Anomaly_Percentage": anomaly_percentage,

            "Training_Time_Seconds": training_time,

            "Testing_Time_Seconds": testing_time,
        })

    # Isolation Forest is used as a reference for agreement only.
    # Jaccard agreement does NOT measure detection accuracy.

    reference = (
        predictions["Isolation Forest"] == -1
    )

    for row in results:

        current = (
            predictions[row["Algorithm"]] == -1
        )

        intersection = np.logical_and(
            reference,
            current,
        ).sum()

        union = np.logical_or(
            reference,
            current,
        ).sum()

        jaccard = (
            intersection / union
            if union > 0
            else np.nan
        )

        row["Agreement_Jaccard_With_Isolation_Forest"] = jaccard

    results_df = pd.DataFrame(results)

    # Preserve the original comparison order.
    # Do not rank algorithms by anomaly percentage or Jaccard agreement.

    return results_df


# --------------------------------------------------
# 4. Save comparison results
# --------------------------------------------------

def save_results(regression_results, anomaly_results):
    """Save Regression and Anomaly Detection comparison results."""

    results_dir = config.PROJECT_DIR / "results"

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    regression_path = (
        results_dir / "regression_comparison.csv"
    )

    anomaly_path = (
        results_dir / "anomaly_comparison.csv"
    )

    regression_results.to_csv(
        regression_path,
        index=False,
    )

    anomaly_results.to_csv(
        anomaly_path,
        index=False,
    )

    print("\nComparison files saved successfully:")

    print(regression_path)

    print(anomaly_path)


# --------------------------------------------------
# 5. Main execution
# --------------------------------------------------

def main():
    """Run both algorithm comparisons and save their results."""

    print("\nLoading processed datasets...")

    X_train, X_test, y_train, y_test = load_data()

    print("\nDATASET INFORMATION")
    print("=" * 60)

    print("Training records:", len(X_train))

    print("Testing records:", len(X_test))

    print("Number of input features:", X_train.shape[1])

    print("Regression target:", config.TARGET_COLUMN)

    print("\nStarting Regression comparison...")

    regression_results = compare_regression(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nStarting Anomaly Detection comparison...")

    anomaly_results = compare_anomaly_detection(
        X_train,
        X_test,
    )

    print("\nREGRESSION COMPARISON RESULTS")
    print("=" * 100)

    print(
        regression_results.round(4).to_string(index=False)
    )

    best_model = regression_results.iloc[0]

    print("\nSELECTED REGRESSION MODEL")
    print("=" * 60)

    print("Selected algorithm:", best_model["Algorithm"])

    print(
        "Cross-Validation MAE:",
        round(best_model["Cross_Validation_MAE"], 4),
    )

    print(
        "Test MAE:",
        round(best_model["Test_MAE"], 4),
    )

    print(
        "Test RMSE:",
        round(best_model["Test_RMSE"], 4),
    )

    print(
        "Test R2:",
        round(best_model["Test_R2"], 4),
    )

    print("\nANOMALY DETECTION COMPARISON RESULTS")
    print("=" * 110)

    print(
        anomaly_results.round(4).to_string(index=False)
    )

    print("\nANOMALY DETECTION EVALUATION NOTE")
    print("=" * 70)

    print(
        "No ground-truth anomaly labels are available in the dataset."
    )

    print(
        "Jaccard agreement measures similarity to Isolation Forest, "
        "not detection accuracy."
    )

    print(
        "Anomaly percentages alone cannot determine which algorithm is best."
    )

    save_results(
        regression_results,
        anomaly_results,
    )

    print("\nAll comparisons completed successfully!")


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    main()
