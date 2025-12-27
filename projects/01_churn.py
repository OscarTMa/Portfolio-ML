import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PAGE CONFIGURATION ---
st.markdown("## 📉 Customer Churn Prediction")
st.markdown("""
This tool uses a specific **XGBoost** model to predict the probability 
of a bank customer leaving the service.
""")

# --- 1. LOAD MODEL & SCALER ---
# We use cache_resource to load these only once, not on every click
@st.cache_resource
def load_artifacts():
    # Paths are relative to the root app.py
    model_path = 'models/churn_xgb_model.pkl'
    scaler_path = 'models/churn_scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    else:
        return None, None

model, scaler = load_artifacts()

# Check if model is loaded
if model is None:
    st.error("⚠️ **Model not found!** Please run the `training_churn.ipynb` notebook first to generate the .pkl files in the 'models' folder.")
    st.stop() # Stop execution here if no model

# --- 2. SIDEBAR INPUTS ---
st.sidebar.header("📝 Customer Profile")

def user_input_features():
    # Demographic Data
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    age = st.sidebar.slider("Age", 18, 92, 30)
    geography = st.sidebar.selectbox("Country", ("France", "Germany", "Spain"))
    
    # Financial Data
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 600)
    tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 3)
    balance = st.sidebar.number_input("Account Balance ($)", 0.0, 250000.0, 60000.0)
    num_of_products = st.sidebar.selectbox("Number of Products", (1, 2, 3, 4))
    has_cr_card = st.sidebar.checkbox("Has Credit Card?", value=True)
    is_active_member = st.sidebar.checkbox("Is Active Member?", value=True)
    estimated_salary = st.sidebar.number_input("Estimated Salary ($)", 0.0, 200000.0, 50000.0)

    # Dictionary with raw data
    data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': 1 if has_cr_card else 0,
        'IsActiveMember': 1 if is_active_member else 0,
        'EstimatedSalary': estimated_salary
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# --- 3. MAIN INTERFACE ---

# Display User Inputs
st.subheader("🔍 Customer Data")
st.dataframe(df, hide_index=True)

# Prediction Logic
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Calculate Churn Risk", type="primary")

with col2:
    if predict_btn:
        with st.spinner('Processing data and predicting...'):
            
            # --- A. PREPROCESSING ---
            # We must replicate the exact steps from the training notebook
            
            df_processed = df.copy()
            
            # 1. Manual Encoding (Label Encoding replication)
            # Gender: Female=0, Male=1
            df_processed['Gender'] = df_processed['Gender'].map({'Female': 0, 'Male': 1})
            
            # Geography: France=0, Germany=1, Spain=2
            df_processed['Geography'] = df_processed['Geography'].map({'France': 0, 'Germany': 1, 'Spain': 2})
            
            # 2. Scaling (Using the loaded StandardScaler)
            # Ensure columns are in the correct order as trained
            feature_order = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                             'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
            
            # Transform data
            input_array = scaler.transform(df_processed[feature_order])
            
            # --- B. PREDICTION ---
            prediction_prob = model.predict_proba(input_array)[0][1] # Probability of Churn (Class 1)
            
            # --- C. DISPLAY RESULTS ---
            st.metric(label="Churn Probability", value=f"{prediction_prob*100:.2f}%")
            
            # Visual Progress Bar
            st.progress(float(prediction_prob))
            
            # Conditional Logic
            if prediction_prob > 0.5:
                st.error("⚠️ **HIGH RISK**: This customer is likely to churn.")
                st.markdown(f"**Recommendation:** Consider offering a retention incentive.")
            else:
                st.success("✅ **LOW RISK**: This customer is likely to stay.")

# --- 4. EXPLANATION TABS ---
st.markdown("---")
tab1, tab2 = st.tabs(["📘 Model Explanation", "📊 Feature Importance"])

with tab1:
    st.markdown("""
    ### How does this model work?
    *(This text should be refined using NotebookLM based on your specific notebook)*
    
    This project uses **XGBoost (Extreme Gradient Boosting)**, a powerful ensemble learning algorithm.
    It combines the predictions of multiple decision trees to produce a highly accurate result.
    
    **Why XGBoost?**
    * It handles tabular data better than Deep Learning in many cases.
    * It provides feature importance scores.
    * It includes regularization to prevent overfitting.
    
    **Handling Imbalance:**
    Since Churn datasets are usually imbalanced (fewer people leave than stay), we used **SMOTE** (Synthetic Minority Over-sampling Technique) during training to teach the model to recognize churners better.
    """)

with tab2:
    st.info("Visual representation of what drives the decision.")
    st.markdown("""
    Based on the training data, the most important factors are usually:
    1. **Age**: Older customers are more volatile.
    2. **Number of Products**: Customers with too many products (3-4) or just 1 often churn.
    3. **Balance**: High balance customers might leave if they find better investment offers elsewhere.
    4. **Active Membership**: Inactive members churn more frequently.
    """)
