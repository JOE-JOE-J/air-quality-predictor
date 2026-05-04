import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model
model = pickle.load(open('best_rf_model.pkl', 'rb'))

st.title("🌫️ California Air Quality Risk Predictor")
st.subheader("Will today be an unsafe air quality day?")

# User inputs
pm25 = st.slider("PM2.5 Concentration", 0.0, 35.0, 8.0)
ozone = st.slider("Ozone Level", 0.0, 0.1, 0.04)
tmax = st.slider("Max Temperature (°F)", 40, 120, 75)
tmin = st.slider("Min Temperature (°F)", 20, 100, 55)
awnd = st.slider("Wind Speed (mph)", 0.0, 30.0, 5.0)
prcp = st.slider("Precipitation (inches)", 0.0, 2.0, 0.0)
population = st.number_input("County Population", 1000, 10000000, 500000)
is_weekend = st.selectbox("Is it a Weekend?", [0, 1])
season = st.selectbox("Season", [1, 2, 3, 4], format_func=lambda x: {1:"Winter",2:"Spring",3:"Summer",4:"Fall"}[x])

temp_range = tmax - tmin
is_rainy = 1 if prcp > 0 else 0
season_2 = 1 if season == 2 else 0
season_3 = 1 if season == 3 else 0
season_4 = 1 if season == 4 else 0

if st.button("Predict Air Quality"):
    input_data = pd.DataFrame([[
        pm25, ozone, is_weekend, population,
        prcp, tmax, tmin, awnd, temp_range, is_rainy,
        season_2, season_3, season_4
    ]], columns=[
        'PM25', 'Ozone', 'IsWeekend', 'Population',
        'PRCP', 'TMAX', 'TMIN', 'AWND', 'Temp_Range', 'Is_Rainy',
        'Season_2', 'Season_3', 'Season_4'
    ])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ UNSAFE — {probability*100:.1f}% probability of unsafe air quality")
    else:
        st.success(f"✅ SAFE — {probability*100:.1f}% probability of unsafe air quality")