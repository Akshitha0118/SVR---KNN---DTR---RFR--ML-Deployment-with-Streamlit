import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Employee Salary Prediction", layout="centered")

# ---------------- LOAD CSS ----------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
dataset = pd.read_csv("emp_sal.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    return {
        "SVR": pickle.load(open("svr_model.pkl", "rb")),
        "KNN": pickle.load(open("knn_model.pkl", "rb")),
        "Decision Tree": pickle.load(open("dt_model.pkl", "rb")),
        "Random Forest": pickle.load(open("rf_model.pkl", "rb"))
    }

models = load_models()

# ---------------- MODEL INFORMATION ----------------
model_info = {
    "SVR": {
        "Algorithm": "Support Vector Regression (Polynomial Kernel)",
        "Kernel": "Polynomial (degree = 4)",
        "C": 6,
        "Gamma": "auto",
        "Epsilon": 1.8
    },
    "KNN": {
        "Algorithm": "K-Nearest Neighbors Regressor",
        "Neighbors": 2,
        "Distance Metric": "Manhattan (p=1)",
        "Weights": "Distance-based"
    },
    "Decision Tree": {
        "Algorithm": "Decision Tree Regressor",
        "Criterion": "Poisson",
        "Max Depth": 3,
        "Random State": 0
    },
    "Random Forest": {
        "Algorithm": "Random Forest Regressor",
        "Trees": 6,
        "Criterion": "Poisson",
        "Max Depth": 4,
        "Random State": 0
    }
}

# ---------------- TITLE ----------------
st.markdown("<h1>💼 Employee Salary Prediction System</h1>", unsafe_allow_html=True)
# ================== MODEL SELECTION ==================
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.markdown("<h2>🧠 Select Machine Learning Model</h2>", unsafe_allow_html=True)

selected_model = st.selectbox(
    "Choose Model",
    list(models.keys())
)



# ================== MODEL INFORMATION ==================
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.markdown("<h2>📘 Model Information</h2>", unsafe_allow_html=True)

info = model_info[selected_model]
for key, value in info.items():
    st.write(f"**{key}:** {value}")

st.markdown("</div>", unsafe_allow_html=True)

# ================== PREDICTION SECTION ==================
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.markdown("<h2>🔮 Salary Prediction</h2>", unsafe_allow_html=True)

level = st.slider("Select Position Level", 1.0, 10.0, 6.5, 0.1)

if st.button("Predict Salary"):
    prediction = models[selected_model].predict([[level]])[0]
    st.markdown(
        f"<div class='result-box'>Predicted Salary: ₹ {prediction:,.2f}</div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# ================== VISUALIZATION SECTION ==================
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.markdown("<h2>📊 Model Visualization</h2>", unsafe_allow_html=True)

plt.figure()
plt.scatter(x, y, label="Actual Data")
plt.plot(x, models[selected_model].predict(x), label="Predicted Curve")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title(f"{selected_model} Model Output")
plt.legend()

st.pyplot(plt)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<center>
📌 Machine Learning Regression Project <br>
SVR • KNN • Decision Tree • Random Forest
</center>
""", unsafe_allow_html=True)
