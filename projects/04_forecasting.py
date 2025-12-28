import streamlit as st
import pandas as pd
import joblib
import os
from prophet.plot import plot_plotly

# --- PAGE CONFIGURATION ---
st.markdown("## 📅 Time Series Forecasting")
st.markdown("""
This tool uses **Meta's Prophet** to predict future sales based on historical trends.
It handles seasonality (e.g., higher sales in summer or holidays) automatically.
""")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    path = 'models/prophet_model.pkl'
    if os.path.exists(path):
        return joblib.load(path)
    return None

model = load_model()

if model is None:
    st.error("⚠️ Model not found! Please run the training notebook first.")
    st.stop()

# --- 2. SIDEBAR CONFIGURATION ---
st.sidebar.header("🔮 Forecast Settings")

horizon = st.sidebar.slider("Months to Predict", min_value=1, max_value=24, value=12)

# --- 3. FORECASTING ---
if st.button("Generate Forecast", type="primary"):
    with st.spinner('Calculating future trends...'):
        
        # A. Create Future Dates
        # Prophet needs a dataframe with future dates to predict on them
        future_dates = model.make_future_dataframe(periods=horizon, freq='MS') # MS = Month Start
        
        # B. Predict
        forecast = model.predict(future_dates)
        
        # --- 4. VISUALIZATION ---
        
        # Metric: Predicted Sales for the last calculated month
        last_prediction = forecast.iloc[-1]
        last_date = last_prediction['ds'].strftime('%B %Y')
        last_value = last_prediction['yhat']
        
        st.subheader(f"Projection for {last_date}")
        st.metric(label="Expected Sales", value=f"{int(last_value):,} units")
        
        # Interactive Plot (Plotly)
        st.subheader("📉 Sales Trend (History + Prediction)")
        fig = plot_plotly(model, forecast)
        
        # Customize the chart to look professional
        fig.update_layout(
            title="Monthly Sales Forecast",
            xaxis_title="Date",
            yaxis_title="Sales Volume",
            hovermode="x"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table (Optional)
        with st.expander("See detailed data"):
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon))

# --- 5. EXPLANATION ---
st.markdown("---")
tab1, tab2 = st.tabs(["📘 How Prophet Works", "📊 Components"])

with tab1:
    st.markdown("""
    ### About the Algorithm
    **Prophet** is an additive regression model designed by Facebook. It decomposes the time series into:
    
    1.  **Trend:** The general direction (sales going up or down long-term).
    2.  **Seasonality:** Repeating patterns (e.g., sales always peak in December).
    3.  **Holidays:** Effects of specific events (not used in this simple demo).
    
    It is robust to missing data and shifts in the trend, making it ideal for business metrics.
    """)

with tab2:
    st.info("Visualizing the 'Seasonality' allows you to see which months are usually the best.")
    # Extract seasonality from the forecast object to show a static insight
    # (In a full app, we would plot the components using model.plot_components(forecast))
    st.markdown("""
    **Insight from Historical Data:**
    * The model has detected a clear **Yearly Seasonality**.
    * Use the interactive chart above to zoom in and see the cyclical patterns.
    """)
