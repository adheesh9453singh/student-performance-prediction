
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "student-mat.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"
RESULT_PATH = BASE_DIR / "reports" / "model_comparison.csv"
OUTPUT_DIR = BASE_DIR / "visualizations"

OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# Load dataset
df = pd.read_csv(DATA_PATH, sep=";")

print("Dataset loaded:", df.shape)

# 1. Final grade distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["G3"], bins=15, kde=True)

plt.title("Distribution of Final Student Grades (G3)")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "grade_distribution.png", dpi=300)
plt.close()


# 2. Correlation heatmap
numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

plt.figure(figsize=(13, 10))
sns.heatmap(
    correlation,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Student Features")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300)
plt.close()


# 3. Model comparison
results = pd.read_csv(RESULT_PATH)

plt.figure(figsize=(10, 6))

results.set_index("Model")[["MAE", "RMSE"]].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Performance Comparison")
plt.ylabel("Error (lower is better)")
plt.xlabel("Machine Learning Model")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=300)
plt.close()


# 4. Random Forest feature importance
model = joblib.load(MODEL_PATH)

estimator = model[-1]
feature_names = model[:-1].get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": estimator.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).head(15)

plt.figure(figsize=(11, 7))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Important Features - Random Forest")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300)
plt.close()

print("\nAll visualizations generated successfully!")

for file in OUTPUT_DIR.iterdir():
    print(file.name)
