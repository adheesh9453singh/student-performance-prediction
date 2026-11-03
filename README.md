cat > README.md <<'EOF'
# Student Performance Prediction System

A Machine Learning application that predicts students' final mathematics grades using student-related information.

## 1. Project Overview

This project uses Python, Pandas, NumPy, Scikit-learn and Streamlit to analyze student data, train predictive models, evaluate their performance and provide predictions through an interactive web application.

## 2. Objectives

- Analyze student academic and personal information.
- Perform data cleaning and exploratory data analysis.
- Train and compare Machine Learning regression algorithms.
- Evaluate models using MAE, RMSE and R².
- Build an interactive student grade prediction application.

## 3. Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook
- Streamlit
- Joblib

## 4. Dataset

The project uses the UCI Student Performance dataset.

- `student-mat.csv`
- `student-por.csv`

The Mathematics dataset is used for the main model training.

- Records: 395
- Columns: 33
- Target variable: G3 (final mathematics grade, 0–20)
- Missing values: 0

## 5. Machine Learning Models

1. Decision Tree Regressor
2. Random Forest Regressor
3. Early Prediction Random Forest (without G1 and G2)

## 6. Model Evaluation Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Decision Tree | 1.28 | 2.34 | 0.73 |
| Random Forest (Full) | 1.18 | 2.00 | 0.80 |
| Random Forest (Early) | 2.97 | 3.75 | 0.31 |

The Full Random Forest achieved the best test-set performance. The Streamlit application uses the Early Prediction model because it does not require G1 and G2 grades.

## 7. Project Structure

- `dataset/` — Student datasets
- `models/` — Saved machine learning models
- `notebooks/` — Jupyter notebooks
- `reports/` — Project reports and evaluation results
- `src/` — Application and visualization scripts
- `visualizations/` — Generated charts

## 8. Visualizations

- Grade distribution
- Correlation heatmap
- Model comparison
- Feature importance

Regenerate charts using:

```bash
python src/generate_visualizations.py