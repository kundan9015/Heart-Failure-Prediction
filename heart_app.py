
import streamlit as st
import pandas as pd
import numpy as np
import pickle
st.set_page_config(page_title="Heart Failure Prediction", page_icon="❤️", layout="wide")

st.title("❤️ Heart Failure Risk Prediction System")
st.write("Enter patient medical data to estimate risk of heart failure event using trained ML model.")
model = pickle.load(open("best_model_rf.pkl","rb"))
scaler = pickle.load(open("scale.pkl","rb"))
features = pickle.load(open("features.pkl","rb"))

st.sidebar.header("Patient Clinical Information")

age = st.sidebar.slider("Age", 18, 100, 50)
anaemia = st.sidebar.selectbox("Anaemia", [0,1])
creatinine_phosphokinase = st.sidebar.number_input("Creatinine Phosphokinase", 10, 8000, 250)
diabetes = st.sidebar.selectbox("Diabetes", [0,1])
ejection_fraction = st.sidebar.slider("Ejection Fraction (%)", 10, 80, 40)
high_blood_pressure = st.sidebar.selectbox("High Blood Pressure", [0,1])
platelets = st.sidebar.number_input("Platelets Count", 10000, 900000, 250000)
serum_creatinine = st.sidebar.number_input("Serum Creatinine", 0.1, 10.0, 1.0)
serum_sodium = st.sidebar.slider("Serum Sodium", 100, 160, 135)
sex = st.sidebar.selectbox("Gender", ["Male","Female"])
smoking = st.sidebar.selectbox("Smoking", [0,1])
time = st.sidebar.slider("Follow-up Period (days)", 1, 400, 100)

input_df = pd.DataFrame({
    "age":[age],
    "anaemia":[anaemia],
    "creatinine_phosphokinase":[creatinine_phosphokinase],
    "diabetes":[diabetes],
    "ejection_fraction":[ejection_fraction],
    "high_blood_pressure":[high_blood_pressure],
    "platelets":[platelets],
    "serum_creatinine":[serum_creatinine],
    "serum_sodium":[serum_sodium],
    "sex":[sex],
    "smoking":[smoking],
    "time":[time]
})


input_df["sex"] = input_df["sex"].map({"Male":1, "Female":0})
st.subheader("Patient Entered Details")
st.dataframe(input_df)
def preprocess(df):

    log_cols = [
        'creatinine_phosphokinase',
        'ejection_fraction',
        'platelets',
        'serum_creatinine',
        'serum_sodium'
    ]

    
    df[log_cols] = np.log1p(df[log_cols])

    
    df[log_cols] = scaler.transform(df[log_cols])

    
    df = df[features]

    return df

processed = preprocess(input_df.copy())

st.sidebar.markdown("---")
with st.sidebar.expander("🛠 Developer Mode (Model Internal Data)"):
    st.write("Processed data after scaling & transformation:")
    st.dataframe(processed)

if st.button("Predict"):

    pred = model.predict(processed)
    prob = model.predict_proba(processed)

    risk = prob[0][1]*100
    safe = prob[0][0]*100

    st.subheader("Diagnosis")

    if pred[0] == 1:
        st.error("🚨 High Risk of Heart Failure")
    else:
        st.success("✅ Patient Stable")

    st.subheader("Probability")
    st.write(f"Safe Probability : {safe:.2f}%")
    st.write(f"Heart Failure Risk : {risk:.2f}%")

    st.subheader("Risk Category")

    if risk < 30:
        st.success("Low Risk")
    elif risk < 60:
        st.warning("Moderate Risk — Doctor Check Recommended")
    else:
        st.error("High Risk — Immediate Medical Attention Needed")

st.markdown("---")
st.markdown("<center><b>Developed by Kundan Kumawat</b></center>", unsafe_allow_html=True)