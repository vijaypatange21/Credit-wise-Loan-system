# Loan Approval Prediction (Credit Project)

This project builds a machine learning model to predict **loan approval** based on applicant and loan-related features.  
The workflow includes **data cleaning, EDA, encoding, correlation analysis, feature engineering, and model training/evaluation**.

The notebook compares **Logistic Regression** and **K-Nearest Neighbors (KNN)**, and the results show that **Logistic Regression performs better** in terms of precision and recall.

---

## Project Files

- `project-credit.ipynb` → Main notebook
- `loan_approval_data.csv` → Dataset used in the notebook (must be in the same folder)

---

## Dataset

The dataset is loaded using:

```python
df = pd.read_csv("loan_approval_data.csv")
```
## Workflow / Steps Covered

### 1) Data Handling (Cleaning)
- Checks missing values  
- Separates categorical and numerical columns  
- Uses `SimpleImputer`:
  - Numerical → filled using **mean**
  - Categorical → filled using **most frequent**

### 2) Exploratory Data Analysis (EDA)
- Basic dataset inspection (`info()`, `head()`)  
- Missing value analysis  

### 3) Encoding
- Converts categorical features into numerical form to support ML models  

### 4) Correlation Heatmap
- Visual correlation analysis to understand relationships between features  

### 5) Train/Test Split
- Splits dataset into training and testing sets for evaluation  

### 6) Model Training and Evaluation
Models used:
- Logistic Regression  
- K-Nearest Neighbors (KNN)  

Evaluation includes:
- Precision score  
- Recall score  
- Comparison between models  

Notebook conclusion:
- KNN performs worse than Logistic Regression  
- Best model: **Logistic Regression**
  - Precision ≈ **~80**
  - Recall ≈ **~80+**

### 7) Feature Engineering
- Additional feature improvements were applied to boost model performance  

---

## Tech Stack
- Python  
- Jupyter Notebook  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

---
