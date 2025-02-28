 My-project

Healthcare Project: Diabetes Prediction

Overview

This project uses Machine Learning to predict diabetes based on patient data.

Problem Statement

Diabetes is a chronic disease that requires early diagnosis to prevent complications.
This project aims to develop an AI model that can efficiently detect diabetes risk.

Dataset

Source: Kaggle - Pima Indians Diabetes Dataset

Features: Age, Blood Pressure, Glucose Level, BMI, etc.


Technologies Used

Python (Data Handling & Modeling)

Pandas, NumPy (Data Processing)

Scikit-Learn (Machine Learning Models)

Matplotlib & Seaborn (Data Visualization)


Approach

1. Data Cleaning & Preprocessing


2. Exploratory Data Analysis (EDA)


3. Model Training & Evaluation


4. Insights & Conclusion



Results

Best Model: Random Forest (Accuracy: 82%)

Key Insight: Glucose levels and BMI are strong indicators of diabetes risk.



Exploratory Data Analysis (EDA)

1. Histogram for Glucose Levels

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

 Load dataset
df = pd.read_csv("diabetes.csv")

 Histogram for glucose levels
plt.figure(figsize=(8, 5))
sns.histplot(df['Glucose'], bins=30, kde=True)
plt.title("Distribution of Glucose Levels")
plt.xlabel("Glucose Level")
plt.ylabel("Count")
plt.show()

2. Correlation Heatmap

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()




Model Evaluation

1. Confusion Matrix

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

 Predictions
y_pred = model.predict(X_test)

 Confusion Matrix
plt.figure(figsize=(5, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

2. Classification Report & ROC-AUC Score

 Classification Report
print(classification_report(y_test, y_pred))

 ROC-AUC Score
roc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"ROC-AUC Score: {roc_score:.2f}")


Visual Results





 Run the Project

1. Clone the Repository

git clone https://github.com/Sirisha-127/Diabetes-Prediction.git
cd Diabetes-Prediction

2. Install Dependencies

pip install -r requirements.txt

3. Run the Model

python main.py


License
MIT License  

 



