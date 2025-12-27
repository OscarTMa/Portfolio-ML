Aquí tienes el código **completo y corregido** para `projects/02_precios.py`.

He incluido la **solución definitiva** para el error de las columnas (`ValueError`). El código ahora le pregunta automáticamente al modelo: *"¿En qué orden quieres los datos?"* y reordena el DataFrame antes de predecir.

Copia y reemplaza todo el contenido de `projects/02_precios.py`:

```python
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
        model = joblib.load(path)
        return model
    return None

model = load_model()

# Check if model exists
if model is None:
    st.error("⚠️ **Model not found!** Please run the `training_prices.ipynb` notebook first to generate the .pkl file.")
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
    # Note: Logic matches the training notebook (Average rooms, not total)
    ave_rooms = st.sidebar.slider("Avg Rooms per Household", 1.0, 10.0, 5.0)
    ave_bedrms = st.sidebar.slider("Avg Bedrooms per Household", 0.5, 5.0, 1.0)
    
    # Dictionary with raw data
    # IMPORTANT: keys must match the training column names exactly
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
    # Streamlit map requires columns named exactly 'lat' and 'lon'
    map_data = pd.DataFrame({'lat': [df['Latitude'][0]], 'lon': [df['Longitude'][0]]})
    st.map(map_data, zoom=6)

with col2:
    st.subheader("💵 Prediction")
    
    st.markdown("<br>", unsafe_allow_html=True) # Spacer
    
    if st.button("Estimate Price", type="primary"):
        with st.spinner('Calculating value...'):
            
            # --- FIX: DYNAMIC REORDERING ---
            # We ask the model explicitly what column order it expects.
            # This prevents the "ValueError: feature_names mismatch" error forever.
            if hasattr(model, 'feature_names_in_'):
                expected_order = model.feature_names_in_
                df_ordered = df[expected_order]
            else:
                df_ordered = df
            
            # Predict
            prediction = model.predict(df_ordered)[0]
            
            # The target in dataset is in units of $100,000, so we multiply
            final_price = prediction * 100000
            
            # Display Metric
            st.metric(label="Estimated Value", value=f"${final_price:,.2f}")
            
            # Context info
            if final_price > 450000:
                st.info("ℹ️ High-value area (Expensive).")
            elif final_price < 200000:
                st.success("ℹ️ Affordable area.")
            else:
                st.info("ℹ️ Mid-range area.")

# Display Input Data Summary
st.markdown("### Selected Parameters")
st.dataframe(df, hide_index=True)

# --- 4. EXPLANATION TABS ---
st.markdown("---")
tab1, tab2 = st.tabs(["📘 Model Logic", "📉 Error Metrics"])

with tab1:
    st.markdown("""
    ### Random Forest Regression
    *(Use NotebookLM to expand on this based on your training notebook)*
    
    This project solves a **Regression** problem (predicting a number).
    
    **How it works:**
    We use a **Random Forest Regressor**, which is a collection of decision trees. 
    Each tree gives an estimate of the price, and the final result is the average of all trees.
    
    **Key Features:**
    1. **Location:** California coastal areas are significantly more expensive.
    2. **Median Income:** Wealthier neighborhoods drive prices up (Strongest correlation).
    3. **House Age:** Older houses in established areas can be more valuable.
    """)

with tab2:
    st.markdown("""
    ### Model Performance
    During training on the California Housing dataset:
    
    * **R² Score:** ~0.80 (The model explains about 80% of price variations).
    * **MAE (Mean Absolute Error):** ~$30,000 - $40,000.
    
    *Note: The model is trained on 1990 census data, so prices reflect that era relative values.*
    """)

```
