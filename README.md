# Predictive Modeling Using Machine Learning

## Project Overview
This project builds machine-learning models to predict students' exam scores from academic and study-related features.

## Objective
Predict **Exam_Score** using:
- Study Hours
- Attendance Percentage
- Assignments Completed
- Previous Score
- Sleep Hours

## Machine Learning Models
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

## Workflow
1. Load the dataset
2. Clean duplicate/missing records
3. Select features and target
4. Split data into training and testing sets
5. Train three regression models
6. Evaluate using MAE, RMSE and R²
7. Generate predictions
8. Visualize model performance and feature importance

## Files
- `student_performance.csv` - dataset
- `predictive_modeling.py` - complete Python implementation
- `outputs/model_results.csv` - model comparison
- `outputs/sample_predictions.csv` - sample predictions
- `outputs/01_exam_score_distribution.png` - score distribution
- `outputs/02_study_hours_vs_exam_score.png` - relationship visualization
- `outputs/03_actual_vs_predicted.png` - prediction performance
- `outputs/04_feature_importance.png` - feature importance
- `requirements.txt` - required Python libraries

## How to Run
Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python predictive_modeling.py
```

## Evaluation Metrics
- **MAE:** average absolute prediction error
- **RMSE:** penalizes larger errors more strongly
- **R²:** proportion of target variation explained by the model

## Expected Outcome
The project demonstrates supervised learning, model training/testing, regression evaluation, prediction, and visualization.
