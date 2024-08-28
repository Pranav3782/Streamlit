import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Title and description
st.title("Heart Disease Prediction App")
st.write("""
This app predicts whether a person has heart disease based on several health parameters.
""")

# Load and display data
st.subheader("Dataset")
heart_data = pd.read_csv('heart_disease_data.csv')
st.write(heart_data.head())

# Data preparation
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model training
model = LogisticRegression()
model.fit(X_train, Y_train)

# Model accuracy
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

st.subheader("Model Accuracy")
st.write(f"Training data accuracy: {training_data_accuracy*100:.2f}%")
st.write(f"Test data accuracy: {test_data_accuracy*100:.2f}%")

# User input
st.subheader("Enter your health parameters:")
age = st.slider("Age", 29, 77, 54)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
trestbps = st.slider("Resting Blood Pressure", 94, 200, 130)
chol = st.slider("Serum Cholesterol (mg/dl)", 126, 564, 250)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"][x])
thalach = st.slider("Maximum Heart Rate Achieved", 71, 202, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.slider("ST depression induced by exercise relative to rest", 0.0, 6.2, 1.0)
slope = st.selectbox("Slope of the peak exercise ST segment", [0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
ca = st.selectbox("Number of major vessels (0-3) colored by flourosopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3], format_func=lambda x: ["Normal", "Fixed defect", "Reversible defect", "Other"][x])

# Prediction
input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

prediction = model.predict(input_data_as_numpy_array)

# Display the prediction result
st.subheader("Prediction")
if prediction[0] == 0:
    st.markdown('<div style="color: green; font-size: 24px;">The Person does not have Heart Disease.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="color: red; font-size: 24px;">The Person has Heart Disease.</div>', unsafe_allow_html=True)
