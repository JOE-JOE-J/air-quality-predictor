import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np

# Load model and scaler
model = pickle.load(open('best_rf_model.pkl', 'rb'))
scaler = joblib.load('scaler.pkl')

st.title("🌫️ California Air Quality Risk Predictor")
st.subheader("Will today's air quality be unsafe (AQI > 75)?")
st.info("💡 This tool can support early warnings for vulnerable populations and public health decision-making.")
st.markdown("---")

# User inputs
pm25 = st.slider("PM2.5 Concentration (µg/m³)", 0.0, 35.0, 7.5)
ozone = st.slider("Ozone Level", 0.0, 0.1, 0.04)
tmax = st.slider("Max Temperature (°F)", 40, 120, 75)
tmin = st.slider("Min Temperature (°F)", 20, 100, 55)
awnd = st.slider("Wind Speed (mph)", 0.0, 30.0, 5.0)
prcp = st.slider("Precipitation (inches)", 0.0, 2.0, 0.0)
population = st.number_input("County Population", 1000, 10000000, 500000)

is_weekend_input = st.selectbox("Is it a Weekend?", ["No", "Yes"])
is_weekend = 1 if is_weekend_input == "Yes" else 0

season = st.selectbox("Season", [1, 2, 3, 4], format_func=lambda x: {1:"Winter",2:"Spring",3:"Summer",4:"Fall"}[x])

# Derived features
temp_range = tmax - tmin
is_rainy = 1 if prcp > 0 else 0
season_2 = 1 if season == 2 else 0
season_3 = 1 if season == 3 else 0
season_4 = 1 if season == 4 else 0

st.markdown("---")

if st.button("🔍 Predict Air Quality"):

    # Apply exact same StandardScaler used during training
    scaled = scaler.transform([[pm25, population]])[0]
    pm25_scaled = scaled[0]
    pop_scaled = scaled[1]

    input_data = pd.DataFrame([[
        pm25_scaled, ozone, is_weekend, pop_scaled,
        prcp, tmax, tmin, awnd, temp_range, is_rainy,
        season_2, season_3, season_4
    ]], columns=[
        'PM25', 'Ozone', 'IsWeekend', 'Population',
        'PRCP', 'TMAX', 'TMIN', 'AWND', 'Temp_Range', 'Is_Rainy',
        'Season_2', 'Season_3', 'Season_4'
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error(f"""
⚠️ **High Risk: Unsafe Air Quality Expected**

**Probability: {probability*100:.1f}%**

Limit outdoor activity, especially for children, the elderly, and those with respiratory conditions.
        """)
    else:
        st.success(f"""
✅ **Low Risk: Air Quality Expected to be Safe**

**Probability of Unsafe Air: {probability*100:.1f}%**

Air quality is expected to be within safe limits today.
        """)

    # Input summary
    st.markdown("---")
    st.markdown("**Input Summary:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PM2.5", f"{pm25} µg/m³")
        st.metric("Ozone", f"{ozone:.3f}")
        st.metric("Wind Speed", f"{awnd} mph")
    with col2:
        st.metric("Max Temp", f"{tmax}°F")
        st.metric("Min Temp", f"{tmin}°F")
        st.metric("Precipitation", f"{prcp} in")
    with col3:
        st.metric("Season", {1:"Winter",2:"Spring",3:"Summer",4:"Fall"}[season])
        st.metric("Weekend", "Yes" if is_weekend else "No")
        st.metric("Population", f"{population:,}")
