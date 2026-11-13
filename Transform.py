
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from Load import load_data
from Clean import clean_data


# 1. Load and clean the dataset
df = load_data()
df = clean_data(df)


# 2. Define numerical columns
numerical_columns = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity"
]


# 3. Define categorical columns
categorical_columns = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Motivation_Level",
    "Internet_Access",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender"
]


# 4. Separate features and target
X = df[numerical_columns + categorical_columns]
y = df["Exam_Score"]


# 5. Split data before fitting transformations
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 6. Apply transformations
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), categorical_columns)
    ]
)


# Fit on training data only
X_train_transformed = preprocessor.fit_transform(X_train)

# Apply the same transformations to test data
X_test_transformed = preprocessor.transform(X_test)


# 7. Convert transformed data into DataFrames
feature_names = preprocessor.get_feature_names_out()

train_df = pd.DataFrame(
    X_train_transformed,
    columns=feature_names,
    index=X_train.index
)

test_df = pd.DataFrame(
    X_test_transformed,
    columns=feature_names,
    index=X_test.index
)

train_df["Exam_Score"] = y_train
test_df["Exam_Score"] = y_test


# 8. Save transformed datasets
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output_dir / "train_transformed.csv", index=False)
test_df.to_csv(output_dir / "test_transformed.csv", index=False)

print("\nData transformation completed successfully!")
print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)
