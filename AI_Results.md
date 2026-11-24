# AI Results Interpretation and Limitations

## 1. Introduction

This section presents the evaluation and interpretation of the Artificial Intelligence models used in the project. Two main tasks were performed: Regression and Classification.

Regression models were used to predict students' final grades, while Classification models were used to categorize student performance into three levels: Low, Medium, and High.

The models were evaluated using different performance metrics to compare their results, identify the best-performing model for each task, and discuss the limitations of the predictions.

---

## 2. Regression Results and Comparison

Two regression models were evaluated:

- Linear Regression
- Random Forest

The models were compared using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²).

### 2.1 Regression Evaluation Results

| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| MAE | 0.416 | 1.067 |
| MSE | 2.314 | 3.870 |
| RMSE | 1.521 | 1.967 |
| R² | 0.825 | 0.707 |

### 2.2 Results Interpretation

Linear Regression performed better than Random Forest across all evaluation metrics.

It achieved lower MAE, MSE, and RMSE values, indicating that its predictions had smaller errors.

Furthermore, Linear Regression achieved an R² score of approximately 0.825, compared to 0.707 for Random Forest. This means that Linear Regression explains approximately 82.5% of the variation in students' final grades.

### 2.3 Best Regression Model

Linear Regression was selected as the best-performing regression model because it achieved lower prediction errors and a higher R² score.

Therefore, it provided more reliable grade predictions than Random Forest in this experiment.

---

## 3. Classification Results and Comparison

Two classification models were evaluated:

- Decision Tree
- Random Forest

These models were used to classify students' performance into Low, Medium, and High categories.

The evaluation included Accuracy, Balanced Accuracy, Precision, Recall, and F1-Score.

### 3.1 Classification Evaluation Results

| Metric | Decision Tree | Random Forest |
|---|---:|---:|
| Accuracy | 97.35% | 98.94% |
| Balanced Accuracy | 49.39% | 49.92% |
| Precision | 49.72% | 55.28% |
| Recall | 49.39% | 49.92% |
| F1-Score | 49.55% | 52.20% |

### 3.2 Results Interpretation

Random Forest achieved better results than Decision Tree across all the reported evaluation metrics.

It achieved an accuracy of 98.94% and an F1-score of 52.20%, compared to 97.35% accuracy and 49.55% F1-score for Decision Tree.

However, both models achieved relatively low Balanced Accuracy and F1-Score values compared to their overall accuracy.

This indicates that the models did not perform equally well across all student performance categories.

### 3.3 Best Classification Model

Random Forest was selected as the better-performing classification model based on its higher overall evaluation scores.

However, its ability to identify minority categories, particularly High, remained limited.

---

## 4. Confusion Matrix Analysis

Confusion matrices were used to examine the classification predictions and identify correct and incorrect classifications for each performance category.

### 4.1 Random Forest Confusion Matrix

| Actual / Predicted | Low | Medium | High |
|---|---:|---:|---:|
| Low | 6 | 6 | 0 |
| Medium | 3 | 1302 | 0 |
| High | 0 | 5 | 0 |

The Random Forest model correctly classified 1,302 students in the Medium category and 6 students in the Low category.

However, it failed to correctly classify any students in the High category. All five actual High cases were incorrectly classified as Medium.

### 4.2 Decision Tree Confusion Matrix

| Actual / Predicted | Low | Medium | High |
|---|---:|---:|---:|
| Low | 6 | 6 | 0 |
| Medium | 6 | 1281 | 18 |
| High | 0 | 5 | 0 |

The Decision Tree model correctly classified 1,281 Medium students and 6 Low students.

However, it also failed to identify any High students. Additionally, 18 Medium students were incorrectly classified as High.

### 4.3 Comparison and Interpretation

Both models performed well in identifying the Medium category but struggled to identify the High category.

Random Forest correctly classified more Medium students and made fewer classification errors in this comparison.

Nevertheless, neither model successfully identified the High category.

These results demonstrate that high overall accuracy does not necessarily indicate balanced classification performance across all categories.

---

## 5. Additional Model Configuration Comparison

Different model configurations and strategies were also evaluated, including class weighting and random oversampling.

The best Random Forest configuration achieved a Macro F1-Score of approximately 52.66% using class weighting.

The highest Balanced Accuracy in the additional configuration comparison was approximately 52.90%, achieved by a Decision Tree configuration using random oversampling.

These results show that different configurations can improve different evaluation metrics. Therefore, model selection should consider more than overall accuracy.

---

## 6. Limitations

Despite the evaluation results, the AI models have several limitations:

### 6.1 Class Imbalance

The classification test data contains a large difference in the number of students in each category. The Medium category has significantly more samples than the Low and High categories.

This imbalance can cause the models to favor the majority category.

### 6.2 Limited Minority-Class Prediction

Both classification models struggled to identify students in the High category.

The Random Forest model did not correctly classify any High cases, which limits its reliability for identifying students in this category.

### 6.3 Prediction Accuracy Limitations

Although the regression models achieved good evaluation results, their predictions are not guaranteed to be completely accurate for every student.

Individual student performance may be affected by factors that are not fully represented in the available dataset.

### 6.4 Dependence on Available Data

The models depend on the quality, distribution, and characteristics of the available dataset.

If new student data has different characteristics, model performance may change.

### 6.5 Generalization Limitations

The evaluation results reflect performance on the available test data. They do not guarantee the same performance on all future student data.

---

## 7. Final Conclusion

The evaluation results showed that Linear Regression was the best-performing model for predicting students' final grades, achieving an R² score of approximately 0.825.

For classification, Random Forest achieved better overall evaluation results than Decision Tree. However, both classification models struggled to identify the High performance category because of the imbalance in the available data.

Overall, the project demonstrates how machine learning models can support student performance prediction and classification. However, the results also highlight the importance of data quality, balanced class distribution, and careful evaluation when developing reliable AI-based prediction systems.