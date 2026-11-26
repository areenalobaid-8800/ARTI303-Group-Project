
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import pandas as pd
from pathlib import Path
import joblib


# 1. Load transformed datasets
train_df = pd.read_csv("data/processed/train_transformed.csv")
test_df = pd.read_csv("data/processed/test_transformed.csv")


# 2. Separate features and target
X_train = train_df.drop(columns=["Exam_Score"])
y_train = train_df["Exam_Score"]

X_test = test_df.drop(columns=["Exam_Score"])
y_test = test_df["Exam_Score"]


# 3. Create performance categories
def classify_performance(score):
    if score < 60:
        return "Low"
    elif score < 80:
        return "Medium"
    else:
        return "High"


y_train_class = y_train.apply(classify_performance)
y_test_class = y_test.apply(classify_performance)


print("\nTraining Class Distribution:")
print(y_train_class.value_counts())

print("\nTesting Class Distribution:")
print(y_test_class.value_counts())


# 4. Create Regression model
regression_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# 5. Train the Regression model
regression_model.fit(X_train, y_train)


# 6. Make Regression predictions
y_pred = regression_model.predict(X_test)


# 7. Evaluate the Regression model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("\nRegression Model Results")
print("------------------------")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)


# 8. Create models directory and save Regression model
output_dir = Path("models")
output_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    regression_model,
    output_dir / "regression_model.pkl"
)

print("\nRegression model saved successfully!")


# 9. Create Classification model
classification_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

# 10. Balance the training data
oversampler = RandomOverSampler(random_state=42)

X_train_balanced, y_train_balanced = oversampler.fit_resample(
    X_train,
    y_train_class
)


print("\nBalanced Training Class Distribution:")
print(y_train_balanced.value_counts())


# 11. Train the Classification model
classification_model.fit(
    X_train_balanced,
    y_train_balanced
)


# 12. Make Classification predictions
y_class_pred = classification_model.predict(X_test)


# 13. Evaluate the Classification model
accuracy = accuracy_score(y_test_class, y_class_pred)

print("\nClassification Model Results")
print("-----------------------------")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test_class,
        y_class_pred,
        labels=["Low", "Medium", "High"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test_class,
    y_class_pred,
    labels=["Low", "Medium", "High"]
))

# 14. Save the Classification model
joblib.dump(
    classification_model,
    output_dir / "classification_model.pkl"
)

print("\nClassification model saved successfully!")
