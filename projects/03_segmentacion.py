import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# --- PAGE CONFIGURATION ---
st.markdown("## 🧩 Customer Segmentation (Clustering)")
st.markdown("""
This tool uses **K-Means Clustering** to group customers based on their Annual Income 
and Spending Score. It helps identify target audiences for marketing campaigns.
""")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    path = 'models/kmeans_model.pkl'
    if os.path.exists(path):
        return joblib.load(path)
    return None

model = load_model()

if model is None:
    st.error("⚠️ Model not found! Please run the training notebook first.")
    st.stop()

# --- 2. SIDEBAR INPUTS ---
st.sidebar.header("👤 New Customer Data")

def user_input():
    income = st.sidebar.slider("Annual Income (k$)", 15, 140, 50)
    score = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)
    
    data = {'Income': income, 'Score': score}
    return pd.DataFrame(data, index=[0])

df_user = user_input()

# --- 3. MAIN INTERFACE ---

# Layout
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📊 Cluster Visualization")
    
    # Generate random background data to visualize the clusters context
    # In a real app, you would load the original dataset training data.
    # Here we simulate 200 points to show the "clouds" of clusters.
    
    # Load centroids from the model
    centroids = model.cluster_centers_
    
    # Predict user cluster
    user_cluster = model.predict(df_user)[0]
    
    # Create a nice Plotly chart
    # We will visualize the Centroids and the User Input
    
    # 1. Create Dataframe for Centroids
    df_centroids = pd.DataFrame(centroids, columns=['Income', 'Score'])
    df_centroids['Type'] = 'Centroid'
    df_centroids['Size'] = 20
    
    # 2. Create Dataframe for User
    df_user['Type'] = 'New Customer'
    df_user['Size'] = 25 # Slightly bigger
    
    # Combine
    df_viz = pd.concat([df_centroids, df_user], ignore_index=True)
    
    # Plot
    fig = px.scatter(
        df_viz, 
        x="Income", 
        y="Score", 
        color="Type",
        size="Size",
        color_discrete_map={'Centroid': 'blue', 'New Customer': 'red'},
        title=f"Customer Position vs Cluster Centers"
    )
    
    # Add background zones (Optional polish)
    fig.add_shape(type="rect", x0=0, y0=0, x1=150, y1=100, 
                  line=dict(color="LightGrey"), fillcolor="White", opacity=0.2)
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🏷️ Result")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Mapping Clusters to Business Names (Based on standard analysis of this dataset)
    # Note: K-Means labels (0,1,2,3,4) can change order if you retrain. 
    # Ideally, we check the centroids to name them dynamically, but for this portfolio 
    # we will use generic names or check the centroid values manually.
    
    # Simple logic based on Income/Score to name the cluster dynamically
    income_val = df_user['Income'][0]
    score_val = df_user['Score'][0]
    
    cluster_name = "Unknown"
    description = ""
    
    # Logic to interpret the result purely based on coordinates (more robust than ID)
    if income_val < 40 and score_val < 40:
        cluster_name = "Sensible Customers"
        description = "Low Income, Low Spending. They save money."
    elif income_val < 40 and score_val > 60:
        cluster_name = "Careless Customers"
        description = "Low Income, High Spending. Risky target."
    elif income_val > 70 and score_val < 40:
        cluster_name = "Miser Customers"
        description = "High Income, Low Spending. Potential for luxury offers."
    elif income_val > 70 and score_val > 60:
        cluster_name = "Target Customers"
        description = "High Income, High Spending. The VIPs."
    else:
        cluster_name = "Standard Customers"
        description = "Medium Income, Medium Spending."

    st.metric(label="Cluster ID", value=str(user_cluster))
    st.info(f"**Segment:** {cluster_name}")
    st.success(f"**Strategy:** {description}")

# --- 4. EXPLANATION ---
st.markdown("---")
st.markdown("""
### 🧠 How K-Means Works


It is an **Unsupervised Learning** algorithm. Unlike the previous projects, we didn't tell the model "This is a VIP customer".
Instead, the model:
1.  Looked at all the data points.
2.  Calculated mathematical distances between them.
3.  Grouped them into 5 distinct "Clusters" (groups) based on similarity.

**Business Value:**
This allows businesses to stop treating everyone the same and start sending specific emails:
* *Discounts* for "Standard Customers".
* *Exclusive products* for "VIPs".
""")
