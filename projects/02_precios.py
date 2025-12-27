import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PAGE CONFIGURATION ---
st.markdown("## 💰 Real Estate Price Prediction")
st.markdown("""
This tool estimates the **median house value** in a California district based on 
demographics and property characteristics using a **Random Forest Regressor**.
""")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    path = 'models/housing_rf_model.pkl'
    if os.path.exists(path):
        return joblib.load(path)
    return None

model = load_model()

if model is None:
    st.error("⚠️ Model not found! Please run the training notebook first.")
    st.stop()

# --- 2. SIDEBAR INPUTS ---
st.sidebar.header("🏡 Property Details")

def user_input_features():
    # Geographical coordinates (Default to Los Angeles area)
    lat = st.sidebar.slider("Latitude", 32.5, 42.0, 34.05)
    lon = st.sidebar.slider("Longitude", -124.3, -114.3, -118.24)
    
    # Demographics
    med_inc = st.sidebar.slider("Median Income (in $10k)", 0.5, 15.0, 5.0)
    house_age = st.sidebar.slider("House Age (Years)", 1, 52, 20)
    population = st.sidebar.slider("Population in Block", 100, 5000, 1000)
    ave_occup = st.sidebar.slider("Avg Occupants per Household", 1.0, 6.0, 3.0)
    
    # Property Specs
    ave_rooms = st.sidebar.slider("Avg Rooms", 1.0, 10.0, 5.0)
    ave_bedrms = st.sidebar.slider("Avg Bedrooms", 0.5, 5.0, 1.0)
    
    # Match column names from training
    data = {
        'MedInc': med_inc,
        'HouseAge': house_age,
        'AveRooms': ave_rooms,
        'AveBedrms': ave_bedrms,
        'Population': population,
        'AveOccup': ave_occup,
        'Latitude': lat,
        'Longitude': lon
    }
    return pd.DataFrame(data, index=[0])

df = user_input_features()

# --- 3. MAIN INTERFACE ---

# Layout: 2 columns (Map and Result)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 Location")
    # Streamlit requires columns named 'lat' and 'lon' for the map
    map_data = pd.DataFrame({'lat': [df['Latitude'][0]], 'lon': [df['Longitude'][0]]})
    st.map(map_data, zoom=6)

with col2:
    st.subheader("💵 Prediction")
    
    if st.button("Estimate Price", type="primary"):
        with st.spinner('Calculating value...'):
            # Predict
            prediction = model.predict(df)[0]
            
            # The target in dataset is in units of $100,000
            final_price = prediction * 100000
            
            st.metric(label="Estimated Value", value=f"${final_price:,.2f}")
            
            # Context info
            if final_price > 400000:
                st.info("ℹ️ This is a high-value area.")
            else:
                st.info("ℹ️ This is an affordable area.")

# Display Input Data Summary
st.markdown("### Selected Parameters")
st.dataframe(df, hide_index=True)

# --- 4. EXPLANATION TABS ---
st.markdown("---")
tab1, tab2 = st.tabs(["📘 Model Logic", "📉 Error Metrics"])

with tab1:
    st.markdown("""
    ### Random Forest Regression
    *(Use NotebookLM to expand on this)*
    
    Unlike the Churn project (Classification), this project solves a **Regression** problem.
    We use a **Random Forest**, which builds multiple decision trees and averages their outputs 
    to predict a continuous numerical value (the price).
    
    **Key Features Used:**
    * **Median Income:** The strongest predictor of house prices.
    * **Location (Lat/Lon):** Crucial for real estate value.
    """)

with tab2:
    st.markdown("""
    ### Model Performance
    During training, the model achieved:
    * **R² Score:** ~0.80 (Explains 80% of price variance)
    * **MAE (Mean Absolute Error):** ~$30,000
    
    *Note: Real estate prices are volatile, and this model is based on 1990s California census data.*
    """)
