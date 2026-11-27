# AI Prediction Module Documentation

## 1. Overview

The AI Prediction module is responsible for loading the trained machine learning models, receiving new student data, and generating student performance predictions.

The module uses two AI tasks:
- **Regression:** Predicts the student's expected exam score (`Exam_Score`).
- **Anomaly Detection:** Identifies unusual patterns in student input data.

## 2. Saved Models

The trained models and preprocessing steps are saved using Joblib in the `models/` directory.

| File | Description |
|---|---|
| `preprocessor.joblib` | Saved preprocessing pipeline used to transform student input data. |
| `regression_model.joblib` | Trained regression model used to predict exam scores. |
| `anomaly_detection_model.joblib` | Trained anomaly detection model used to identify unusual patterns. |

## 3. Input Data

The prediction module accepts student data as a Python dictionary or a Pandas DataFrame.

Each student input must contain 19 features:
- 6 numerical features.
- 13 categorical features.

The input validation function checks:
- Missing or unexpected columns.
- Numerical data types and allowed ranges.
- Valid categorical values.
- Empty or multiple-student inputs.

Invalid inputs raise clear errors before prediction.

## 4. Model Loading

The `StudentPerformancePredictor` class loads the saved preprocessor and trained models using Joblib.

It also checks model compatibility, including the number and names of transformed features.

## 5. Prediction Process

The `predict()` method performs the following steps:

1. Validates the new student's input data.
2. Transforms the input using the saved preprocessor.
3. Predicts the expected exam score using the regression model.
4. Detects unusual input patterns using the anomaly detection model.
5. Returns the prediction results.

## 6. Prediction Outputs

The prediction function returns a dictionary containing:

| Output | Description |
|---|---|
| `predicted_exam_score` | Predicted exam score. |
| `anomaly_prediction` | Anomaly detection result: 1 for Normal and -1 for Anomaly. |
| `anomaly_status` | Human-readable anomaly status: Normal or Anomaly. |
| `anomaly_score` | Decision function score produced by the anomaly detection model. |

## 7. How to Run

Activate the project's virtual environment and run the following command from the project directory:

```bash
python AI_Prediction.py
```

To load the saved models and perform a prediction, use:

```python
from AI_Prediction import StudentPerformancePredictor

predictor = StudentPerformancePredictor()

results = predictor.predict(student_data)

print(results)
```

Replace `student_data` with a valid dictionary containing the 19 required student features.

## 8. Testing

The prediction module was tested using Pytest.

Run the automated tests with:

```bash
python -m pytest test_ai_prediction.py -v
```

**Testing Results:**
- Total tests: 7
- Passed: 7
- Failed: 0
- Execution time: 3.52 seconds

All seven tests passed successfully, covering input validation, model loading, prediction functionality, and prediction score range.

## 9. Conclusion

The AI Prediction module successfully loads the saved machine learning models, validates new student data, applies the saved preprocessing pipeline, predicts exam scores, and detects unusual input patterns.

The module is ready to be integrated into the Student Performance Prediction (SPP) application, subject to final integration testing.