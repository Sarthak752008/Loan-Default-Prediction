# Loan-Default-Prediction

# 🏦 Loan Approval Prediction using Random Forest Classifier

This project aims to predict loan approval outcomes based on applicant details using machine learning. A **Random Forest Classifier** is applied for classification analysis on real-world loan data.

---

## 📌 Problem Statement

Loan approval is a critical process in banking, requiring accurate evaluation of multiple parameters such as income, credit history, education, and employment. This project automates the approval prediction using a supervised machine learning approach to help financial institutions make more efficient and accurate decisions.

---

## 🧾 Dataset Overview

- **Source**: Public dataset (`train.csv`)
- **Total Samples**: 614
- **Target Variable**: `Loan_Status` (Y: Approved, N: Not Approved)
- **Features**: Gender, Marital Status, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area

---

## 🛠️ Methodology

1. **Data Preprocessing**:
   - Dropped `Loan_ID` as it's a non-predictive identifier.
   - Removed rows with missing values (for simplicity).
   - Encoded categorical variables using `LabelEncoder`.
   - Scaled numerical features using `StandardScaler`.

2. **Model Development**:
   - Performed a train-test split (80/20) with stratified sampling.
   - Trained a `RandomForestClassifier` using the training data.
   - Made predictions and computed performance metrics.

3. **Evaluation Metrics**:
   - Confusion Matrix with counts and percentage
   - Accuracy, Precision, Recall, F1-Score
   - ROC Curve & AUC Score
   - Feature Importance Plot

---

## 📈 Results

| Metric        | Value    |
|---------------|----------|
| Accuracy      | ~0.80    |
| Precision     | ~0.84    |
| Recall        | ~0.87    |
| AUC Score     | ~0.84    |

**Top Important Features**:
- `Credit_History`
- `ApplicantIncome`
- `LoanAmount`

---

## 📊 Visualizations

- 📌 Confusion Matrix (with percentage)
- 📈 ROC Curve
- 🌟 Feature Importance Bar Chart

---

## 🚀 Installation

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
