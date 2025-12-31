import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Employee Salary Predictor",
    layout="centered"
)

# ---------------- LOAD CSS ----------------
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv('emp_sal.csv')   # keep CSV in same folder
    x = df.iloc[:, 1:2].values
    y = df.iloc[:, 2].values
    return x, y

x, y = load_data()

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    with open("svr_model.pkl", "rb") as f:
        svr = pickle.load(f)
    with open("knn_model.pkl", "rb") as f:
        knn = pickle.load(f)
    with open("dt_model.pkl", "rb") as f:
        dt = pickle.load(f)
    with open("rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    return svr, knn, dt, rf

svr, knn, dt, rf = load_models()

# ---------------- UI ----------------
st.markdown('<div class="title">💼 Employee Salary Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict salary using ML models with visualization</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

# Model selection
model_name = st.selectbox(
    "Select ML Model",
    ["SVR", "KNN", "Decision Tree", "Random Forest"]
)

# Input
level = st.number_input(
    "Enter Position Level",
    min_value=1.0,
    max_value=10.0,
    step=0.1,
    value=6.5
)

# Select model
if model_name == "SVR":
    model = svr
elif model_name == "KNN":
    model = knn
elif model_name == "Decision Tree":
    model = dt
else:
    model = rf

# Prediction
if st.button("Predict Salary"):
    x_input = np.array([[level]])
    prediction = model.predict(x_input)[0]

    st.markdown(
        f'<div class="result">💰 Predicted Salary: ₹ {prediction:,.2f}</div>',
        unsafe_allow_html=True
    )

    # ---------------- VISUALIZATION ----------------
    st.subheader("📈 Model Visualization")

    x_grid = np.arange(min(x), max(x), 0.1).reshape(-1, 1)
    y_pred_curve = model.predict(x_grid)

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(x_grid, y_pred_curve)
    ax.set_xlabel("Position Level")
    ax.set_ylabel("Salary")
    ax.set_title(f"{model_name} Regression Curve")

    st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
    "<hr><center>Built with ❤️ using Streamlit & Machine Learning</center>",
    unsafe_allow_html=True
)

