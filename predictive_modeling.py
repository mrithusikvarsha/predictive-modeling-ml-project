"""
Predictive Modeling Using Machine Learning
Student Performance Prediction Project
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "student_performance.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load data
df = pd.read_csv(DATA_FILE)

# 2. Basic data cleaning
df = df.drop_duplicates()
df = df.dropna()

# 3. Features and target
features = [
    "Study_Hours",
    "Attendance_Percent",
    "Assignments_Completed",
    "Previous_Score",
    "Sleep_Hours"
]
X = df[features]
y = df["Exam_Score"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 5. Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=8, random_state=42
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2_Score": round(r2, 3)
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_results.csv"), index=False)
print("\nMODEL PERFORMANCE")
print(results_df.to_string(index=False))

# 6. Prediction example
final_model = models["Random Forest"]
sample = X_test.head(5)
sample_predictions = final_model.predict(sample)

prediction_df = sample.copy()
prediction_df["Actual_Exam_Score"] = y_test.loc[sample.index].values
prediction_df["Predicted_Exam_Score"] = sample_predictions.round(2)
prediction_df.to_csv(
    os.path.join(OUTPUT_DIR, "sample_predictions.csv"), index=False
)

# 7. Actual vs predicted plot
plt.figure(figsize=(7, 5))
plt.scatter(y_test, sample_predictions if len(y_test) == len(sample_predictions) else final_model.predict(X_test), alpha=0.7)
pred_all = final_model.predict(X_test)
plt.clf()
plt.scatter(y_test, pred_all, alpha=0.7)
low = min(y_test.min(), pred_all.min())
high = max(y_test.max(), pred_all.max())
plt.plot([low, high], [low, high])
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Actual vs Predicted Exam Scores - Random Forest")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_actual_vs_predicted.png"), dpi=160)
plt.close()

# 8. Feature importance
importance = pd.Series(
    final_model.feature_importances_, index=features
).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
importance.sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_feature_importance.png"), dpi=160)
plt.close()

importance.to_csv(
    os.path.join(OUTPUT_DIR, "feature_importance.csv"),
    header=["Importance"]
)

print("\nProject completed. Results saved in the outputs folder.")
