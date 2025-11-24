import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("Sistema de Detección Temprana de ACV")

st.write("Aplicación demostrativa para estimar el riesgo de accidente cerebrovascular.")

# Cargar dataset
data = pd.read_csv("stroke_data.csv")

# Seleccionar columnas importantes (ajusta si usas otras)
X = data[["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]]
y = data["stroke"]

# Entrenar modelo rápido solo para la demo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.subheader("Ingresar datos del paciente")

age = st.number_input("Edad", 0, 120, 45)
glucose = st.number_input("Nivel de glucosa", 0.0, 300.0, 110.0)
bmi = st.number_input("BMI", 0.0, 60.0, 25.0)
hypertension = st.selectbox("Hipertensión", ["No", "Sí"])
heart_disease = st.selectbox("Enfermedad cardíaca", ["No", "Sí"])

hypertension_val = 1 if hypertension == "Sí" else 0
heart_disease_val = 1 if heart_disease == "Sí" else 0

if st.button("Predecir"):
    input_data = [[age, glucose, bmi, hypertension_val, heart_disease_val]]
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.error(f"⚠ Riesgo ALTO de ACV (prob = {proba:.2f})")
    else:
        st.success(f"✔ Riesgo BAJO de ACV (prob = {proba:.2f})")

