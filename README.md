 My-project
 AI Healthcare Project: Diabetes Prediction  

  Overview  
This project uses Machine Learning to predict diabetes based on patient data.  

  Problem Statement  
Diabetes is a chronic disease that needs early diagnosis to prevent complications.  
This project aims to develop an AI model that can help detect diabetes risk efficiently.  

 Dataset  
- Source: Kaggle (Diabetes Dataset)  
- Features: Age, Blood Pressure, Glucose Level, etc.  

 Technologies Used  
- Python, Pandas, NumPy  
- Scikit-Learn for ML models  
- Matplotlib & Seaborn for visualization  

Approach  
1. Data Cleaning & Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Model Training & Evaluation  
4. Insights & Conclusion  

 Results  
- Best Model: Random Forest (Accuracy: 82%)  
- Key Insight: Glucose levels and BMI are strong indicators of diabetes risk.  

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Load dataset
df = pd.read_csv("diabetes.csv")

 Histogram for glucose levels
plt.figure(figsize=(8, 5))
sns.histplot(df['Glucose'], bins=30, kde=True)
plt.title("Distribution of Glucose Levels")
plt.show()

 Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns

Predictions
y_pred = model.predict(X_test)

 Confusion Matrix
plt.figure(figsize=(5, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

 Classification Report
print(classification_report(y_test, y_pred))

ROC-AUC Score
roc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"ROC-AUC Score: {roc_score:.2f}")
![Glucose Distribution](images/glucose_distribution.png)  
![Confusion Matrix](images/confusion_matrix.png)

