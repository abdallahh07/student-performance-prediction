# Student Performance Prediction

## Problem
Predicting students' final exam scores based on study habits, attendance, and demographic data.

## Dataset
- Source: student_performance.csv
- Features: study hours, attendance rate, age, gender, internet access, parental education
- Target: final_score

## Models Used
- Linear Regression
- Decision Tree
- Random Forest
- XGBoost

## Results
| Model | R2 Score |
|---|---|
| Linear Regression | 0.748 |
| Decision Tree | 0.459 |
| Random Forest | 0.717 |
| XGBoost | 0.671 |

**Best Model: Linear Regression (R2 = 0.748)**

## Libraries
- scikit-learn, pandas, numpy, matplotlib, seaborn, xgboost

## What I Learned
- Building end-to-end ML pipelines with sklearn
- Handling missing values with SimpleImputer
- Encoding categorical features with OneHotEncoder
- Comparing multiple regression models
