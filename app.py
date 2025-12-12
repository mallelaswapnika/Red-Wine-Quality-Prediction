import streamlit as st
import numpy as np
import joblib

# ============================
# 1. Load model and scaler
# ============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ============================
# 2. Page config
# ============================
st.set_page_config(
    page_title="Red Wine Quality Prediction",
    layout="centered"
)

st.title("🍷 Red Wine Quality Prediction")
st.write(
    "Enter the **chemical properties** of red wine below and click "
    "**Predict Quality** to see the model’s prediction (0–10)."
)

st.markdown("---")

# ============================
# 3. Input form
#    Order MUST match training:
#    fixed acidity, volatile acidity, citric acid, residual sugar,
#    chlorides, free sulfur dioxide, total sulfur dioxide,
#    density, pH, sulphates, alcohol
# ============================

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input(
        "Fixed Acidity (g(tartaric acid)/dm³)",
        min_value=4.0, max_value=16.0, value=8.0, step=0.1
    )
    volatile_acidity = st.number_input(
        "Volatile Acidity (g(acetic acid)/dm³)",
        min_value=0.10, max_value=1.60, value=0.50, step=0.01
    )
    citric_acid = st.number_input(
        "Citric Acid (g/dm³)",
        min_value=0.00, max_value=1.00, value=0.30, step=0.01
    )
    residual_sugar = st.number_input(
        "Residual Sugar (g/dm³)",
        min_value=0.5, max_value=15.0, value=2.5, step=0.1
    )
    chlorides = st.number_input(
        "Chlorides (g(sodium chloride)/dm³)",
        min_value=0.010, max_value=0.500, value=0.080, step=0.001
    )

with col2:
    free_sulfur_dioxide = st.number_input(
        "Free Sulfur Dioxide (mg/dm³)",
        min_value=1, max_value=80, value=15, step=1
    )
    total_sulfur_dioxide = st.number_input(
        "Total Sulfur Dioxide (mg/dm³)",
        min_value=6, max_value=300, value=46, step=1
    )
    density = st.number_input(
        "Density (g/cm³)",
        min_value=0.9900, max_value=1.0050, value=0.9968,
        step=0.0001, format="%.4f"
    )
    pH = st.number_input(
        "pH",
        min_value=2.50, max_value=4.50, value=3.30, step=0.01
    )
    sulphates = st.number_input(
        "Sulphates (g(potassium sulphate)/dm³)",
        min_value=0.30, max_value=2.00, value=0.60, step=0.01
    )
    alcohol = st.number_input(
        "Alcohol (% vol)",
        min_value=8.0, max_value=15.0, value=10.0, step=0.1
    )

st.markdown("---")

# ============================
# 4. Prediction
# ============================
if st.button("🔮 Predict Quality"):
    # Put inputs into correct shape (1, 11)
    input_data = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]])

    # Scale using same scaler as training
    input_scaled = scaler.transform(input_data)

    # Predict
    pred = model.predict(input_scaled)[0]
    quality_pred = round(float(pred), 2)

    st.subheader(f"⭐ Predicted Wine Quality: **{quality_pred} / 10**")

    # Simple interpretation
    if quality_pred >= 7:
        st.success("This looks like a **high-quality** wine! 🍷")
    elif quality_pred >= 5:
        st.info("This seems like an **average-quality** wine 🙂")
    else:
        st.warning("This seems like a **low-quality** wine 😕")

st.caption("Model: RandomForestRegressor trained on UCI Red Wine Quality dataset.")
