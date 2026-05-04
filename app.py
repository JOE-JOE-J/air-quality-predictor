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

col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5 Concentration (µg/m³)", min_value=0.0, max_value=35.0, value=7.5, step=0.1)
    ozone_ppb = st.number_input("Ozone Level (ppb)", min_value=0, max_value=100, value=40, step=1)
    tmax = st.number_input("Max Temperature (°F)", min_value=40, max_value=120, value=75, step=1)
    tmin = st.number_input("Min Temperature (°F)", min_value=20, max_value=100, value=55, step=1)

with col2:
    awnd = st.number_input("Wind Speed (mph)", min_value=0.0, max_value=30.0, value=5.0, step=0.1)
    prcp = st.number_input("Precipitation (inches)", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    population = st.number_input("County Population", min_value=1000, max_value=10000000, value=500000, step=1000)
    season = st.selectbox("Season", [1, 2, 3, 4], format_func=lambda x: {1:"Winter",2:"Spring",3:"Summer",4:"Fall"}[x])

is_weekend_input = st.selectbox("Is it a Weekend?", ["No", "Yes"])
is_weekend = 1 if is_weekend_input == "Yes" else 0

st.markdown("---")

if st.button("🔍 Predict Air Quality"):

    # Convert ozone from ppb back to ppm for model
    ozone = ozone_ppb / 1000.0

    # Derived features
    temp_range = tmax - tmin
    is_rainy = 1 if prcp > 0 else 0
    season_2 = 1 if season == 2 else 0
    season_3 = 1 if season == 3 else 0
    season_4 = 1 if season == 4 else 0

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
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("PM2.5", f"{pm25} µg/m³")
        st.metric("Ozone", f"{ozone_ppb} ppb")
        st.metric("Wind Speed", f"{awnd} mph")
    with c2:
        st.metric("Max Temp", f"{tmax}°F")
        st.metric("Min Temp", f"{tmin}°F")
        st.metric("Precipitation", f"{prcp} in")
    with c3:
        st.metric("Season", {1:"Winter",2:"Spring",3:"Summer",4:"Fall"}[season])
        st.metric("Weekend", "Yes" if is_weekend else "No")
        st.metric("Population", f"{population:,}")
