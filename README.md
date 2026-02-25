# ❤️ Heart Failure Risk Prediction (Machine Learning Web App)

## 🚀 Live Demo
https://heart-failure-prediction-bpnbtvdy8hceuxe8sqzbiq.streamlit.app

A Machine Learning based web application that predicts the risk of heart failure in a patient using clinical health parameters.  
The model is deployed as an interactive web app using **Streamlit**, where users can input medical values and instantly receive prediction results.

---

## 📌 Problem Statement
Heart failure is a serious medical condition and early detection can help doctors take preventive action.

This project predicts whether a patient is at **high risk or low risk of heart failure** using clinical medical attributes such as age, ejection fraction, serum creatinine and sodium level.

---

## 🧠 Machine Learning Approach
The model was trained on the **Heart Failure Clinical Records Dataset**.

Steps performed:
- Data Cleaning
- Handling missing values
- Feature Engineering
- Log Transformation
- Feature Scaling (StandardScaler)
- Model Training & Evaluation

---

## 🤖 Model Used
**Random Forest Classifier**

Why Random Forest?
- Handles non-linear relationships
- Robust to noise
- Works well for tabular medical datasets

---

## 📊 Model Performance
Test F1 Score: **0.76**

The difference between training and testing accuracy indicates slight overfitting, which is common in medical datasets due to limited samples.  
To reduce this, preprocessing and hyperparameter tuning were applied.

---

## 🖥️ Web Application Features
- Interactive user interface
- Real time prediction
- Probability estimation
- Risk categorization (Low / Moderate / High)
- Hidden developer mode showing processed features

---

## 📥 Input Features
The model takes following patient information:

- Age
- Anaemia
- Creatinine Phosphokinase
- Diabetes
- Ejection Fraction
- High Blood Pressure
- Platelets
- Serum Creatinine
- Serum Sodium
- Sex
- Smoking
- Follow-up time (days)

---

## ⚙️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle

---
