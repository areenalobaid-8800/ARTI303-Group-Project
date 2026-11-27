# AI Results Analysis and Model Comparison

## 1. Overview

This section evaluates and compares different algorithms for the two AI tasks in the Student Performance Prediction System (SPP):

1. **Regression:** Predicting the numerical `Exam_Score` using student-related input features.
2. **Anomaly Detection:** Identifying unusual patterns in student input features using an unsupervised learning approach.

All algorithms were evaluated using the same processed training and testing datasets to ensure a consistent comparison.

## 2. Dataset and Evaluation Setup

The processed dataset contains:

- Training records: 5,284
- Testing records: 1,322
- Transformed input features: 40
- Regression target: `Exam_Score`
- Train/test split: 80% training and 20% testing

The same training and testing split was used for all comparisons.

For anomaly detection, `Exam_Score` was excluded from the input features to prevent the target variable from influencing anomaly identification.

## 3. Regression Model Comparison

Four regression algorithms were evaluated:

- Random Forest Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- Dummy Regressor (mean baseline)

### 3.1 Evaluation Metrics

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted scores. Lower values are better.
- **RMSE (Root Mean Squared Error):** Measures prediction error while penalizing larger errors more strongly. Lower values are better.
- **R² (Coefficient of Determination):** Measures how well the model explains variation in the target values. Higher values are generally better.
- **Cross-Validation MAE:** Measures the average MAE across five training folds. Lower values indicate better cross-validation performance.

### 3.2 Comparison Results

The models were evaluated using 5-Fold Cross-Validation on the training dataset. Each model was then trained on the complete training dataset and evaluated on the held-out test dataset.

| Algorithm | CV MAE | CV MAE Std | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.8500 | 0.0411 | 0.8246 | 2.1737 | 0.7041 |
| Random Forest | 1.1322 | 0.0222 | 1.1162 | 2.3539 | 0.6530 |
| Decision Tree | 1.8450 | 0.0354 | 1.7269 | 3.3487 | 0.2977 |
| Dummy Baseline | 2.8254 | 0.0639 | 2.9067 | 3.9973 | -0.0006 |

### 3.3 Execution Times

The following table presents the execution times recorded during the comparison. These times may vary depending on the computer and execution environment.

| Algorithm | CV Time (seconds) | Training Time (seconds) | Testing Time (seconds) |
|---|---:|---:|---:|
| Gradient Boosting | 3.9819 | 0.7893 | 0.0028 |
| Random Forest | 8.0325 | 1.2047 | 0.1114 |
| Decision Tree | 0.2604 | 0.0523 | 0.0018 |
| Dummy Baseline | 0.0181 | 0.0005 | 0.0000 |

### 3.4 Results Interpretation

Gradient Boosting achieved the best regression results among the evaluated algorithms.

It obtained the lowest Cross-Validation MAE (0.8500), the lowest Test MAE (0.8246) and RMSE (2.1737), as well as the highest R² (0.7041).

Compared with Random Forest, Gradient Boosting reduced Test MAE by approximately 26.1% and improved R² from 0.6530 to 0.7041.

Random Forest also performed better than the Decision Tree and Dummy Baseline across all three test evaluation metrics.

The Dummy Regressor provides a simple reference point by predicting the mean training score. The results show that the evaluated machine learning models provide better predictions than this baseline.

### 3.5 Selected Regression Model

**Selected Model: Gradient Boosting Regressor**

Gradient Boosting was selected because it achieved the lowest Cross-Validation MAE and the strongest results across the test evaluation metrics.

It provided lower prediction errors and a higher R² than the other evaluated models.

Therefore, it is the best-performing regression algorithm in this comparison.

**Implementation Status:**

The project implementation has been updated to use Gradient Boosting Regressor as the selected regression model instead of Random Forest.

The regression training and evaluation modules have been updated accordingly, and their unit tests have passed successfully.

The saved regression model must be retrained using the final implementation to ensure that the application uses the selected Gradient Boosting model.

## 4. Anomaly Detection Model Comparison

Three unsupervised anomaly detection algorithms were evaluated:

- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM

All algorithms were trained using the same training features and evaluated on the same testing features.

The `Exam_Score` target was excluded from anomaly detection inputs.

The contamination setting was 0.05 for Isolation Forest and Local Outlier Factor. One-Class SVM used `nu=0.05`.

### 4.1 Comparison Results

| Algorithm | Normal Count | Anomaly Count | Anomaly Percentage | Jaccard Agreement with Isolation Forest |
|---|---:|---:|---:|---:|
| Isolation Forest | 1,256 | 66 | 4.9924% | 1.0000 |
| Local Outlier Factor | 1,254 | 68 | 5.1437% | 0.2523 |
| One-Class SVM | 1,239 | 83 | 6.2784% | 0.1920 |

### 4.2 Execution Times

| Algorithm | Training Time (seconds) | Testing Time (seconds) |
|---|---:|---:|
| Isolation Forest | 0.6202 | 0.0331 |
| Local Outlier Factor | 0.1689 | 0.0406 |
| One-Class SVM | 0.1694 | 0.0515 |

### 4.3 Results Interpretation

Isolation Forest identified 66 unusual observations, representing approximately 4.99% of the testing dataset.

Local Outlier Factor identified 68 unusual observations, while One-Class SVM identified 83.

The Jaccard agreement values describe how much the detected anomaly sets overlap with the Isolation Forest results.

Local Outlier Factor achieved a Jaccard agreement of 0.2523, while One-Class SVM achieved 0.1920.

These values indicate that the algorithms identify different sets of observations. A higher agreement does not necessarily mean higher detection accuracy.

The observed anomaly percentage for Isolation Forest is close to its configured contamination value of 0.05. This is consistent with the model setting, but it does not independently prove that the detected observations are genuine anomalies.

### 4.4 Selected Anomaly Detection Model

**Selected Model: Isolation Forest**

Isolation Forest was retained as the selected anomaly detection model for the current system.

It is already used in the project implementation, and its testing results are consistent with the configured contamination level.

However, the comparison does not establish that Isolation Forest is objectively more accurate than the other algorithms because the dataset does not contain ground-truth anomaly labels.

Its selection should therefore be considered a practical project choice rather than a proven accuracy ranking.

## 5. Limitations

The results should be interpreted with the following limitations:

1. **Dataset limitations:** The models depend on the available dataset features and may not represent every factor that affects student academic performance.

2. **Prediction uncertainty:** Predicted exam scores are estimates based on learned patterns. They do not guarantee an individual student's actual future score.

3. **Anomaly validation:** The dataset does not provide ground-truth anomaly labels. Therefore, anomaly detection accuracy, precision, recall, and F1-score cannot be established from this evaluation.

4. **Algorithm comparison:** Regression metrics provide a direct basis for comparing prediction performance on the test set. In contrast, anomaly detection counts and agreement scores describe model behavior but do not establish which algorithm detects the correct anomalies.

5. **Generalization:** Performance on this dataset does not guarantee identical results for other schools, student populations, or future academic settings.

6. **Decision support:** The system is intended to support teachers by providing predictions and identifying unusual input patterns. It should not replace teachers' professional judgment.

An anomaly indicates an unusual combination of input features. It does not necessarily indicate poor academic performance, incorrect data, or a problem with the student.

## 6. Conclusion

The comparison showed that Gradient Boosting achieved the best regression performance, with a Cross-Validation MAE of 0.8500, Test MAE of 0.8246, RMSE of 2.1737, and R² of 0.7041.

For anomaly detection, Isolation Forest was retained as the selected model for the current system. However, its detection performance cannot be objectively ranked against the other algorithms without labeled anomaly data.

Overall, the comparison provides an evidence-based assessment of the evaluated algorithms and highlights the importance of considering model limitations when interpreting predictions and unusual patterns.

## 7. Generated Comparison Files

The comparison script generated the following files:

- `results/regression_comparison.csv`
- `results/anomaly_comparison.csv`

These files contain the numerical results used in this analysis.