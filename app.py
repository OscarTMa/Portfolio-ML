import streamlit as st

# Configuración de la página (título y layout)
st.set_page_config(page_title="Mi Portafolio ML/AI", layout="wide")

# Introducción o Página de Inicio
def intro():
    st.write("# 👋 Bienvenido a mi Portafolio de ML")
    st.markdown("""
    Este portafolio contiene 5 proyectos clave de Machine Learning e IA, 
    cubriendo desde modelos de regresión hasta NLP.
    
    👈 **Selecciona un proyecto en el menú lateral para comenzar.**
    
    ### Tech Stack:
    - **Python** (Scikit-learn, Pandas, Prophet, Transformers)
    - **Streamlit** (Frontend)
    - **GitHub** (Control de versiones)
    """)

# Definición de las páginas
# Nota: 'projects/01_churn.py' es la ruta a tus archivos
pg = st.navigation([
    st.Page(intro, title="Inicio", icon="🏠"),
    st.Page("projects/01_churn.py", title="1. Churn Prediction", icon="📉"),
    st.Page("projects/02_precios.py", title="2. Predicción de Precios", icon="💰"),
    st.Page("projects/03_segmentacion.py", title="3. Segmentación (Clustering)", icon="🧩"),
    st.Page("projects/04_forecasting.py", title="4. Series Temporales", icon="📅"),
    st.Page("projects/05_nlp.py", title="5. NLP Classifier", icon="🤖"),
])

# Ejecutar la navegación
pg.run()
