import streamlit as st

# Page Configuration
st.set_page_config(page_title="My ML/AI Portfolio", layout="wide")

# Introduction / Home Page
def intro():
    st.write("# 👋 Welcome to my ML Portfolio")
    st.markdown("""
    This portfolio features **5 key Machine Learning & AI projects**, 
    demonstrating skills ranging from Regression and Classification to NLP and Time Series Forecasting.
    
    👈 **Select a project from the sidebar to explore.**
    
    ### 🛠️ Tech Stack:
    - **Core:** Python (Scikit-learn, Pandas, NumPy)
    - **Advanced:** XGBoost, Prophet, Transformers (Hugging Face)
    - **Frontend:** Streamlit
    - **Version Control:** GitHub
    """)

# Page Navigation Setup
# Note: Ensure the 'projects' folder exists with these .py files inside
pg = st.navigation([
    st.Page(intro, title="Home", icon="🏠"),
    st.Page("projects/01_churn.py", title="1. Churn Prediction", icon="📉"),
    st.Page("projects/02_precios.py", title="2. Price Prediction", icon="💰"),
    st.Page("projects/03_segmentacion.py", title="3. User Segmentation", icon="🧩"),
    st.Page("projects/04_forecasting.py", title="4. Time Series", icon="📅"),
    st.Page("projects/05_nlp.py", title="5. NLP Classifier", icon="🤖"),
])

# Run Navigation
pg.run()
