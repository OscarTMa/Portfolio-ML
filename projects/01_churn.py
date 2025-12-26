import streamlit as st
import pandas as pd
import numpy as np
import time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.markdown("## 📉 Predicción de Abandono de Clientes (Churn)")
st.markdown("""
Esta herramienta utiliza un modelo de Machine Learning para predecir la probabilidad 
de que un cliente bancario abandone el servicio. Ajusta los parámetros abajo para simular un perfil.
""")

# --- BARRA LATERAL (SIDEBAR) PARA INPUTS ---
st.sidebar.header("📝 Perfil del Cliente")

def user_input_features():
    # Datos Demográficos
    gender = st.sidebar.selectbox("Género", ("Masculino", "Femenino"))
    age = st.sidebar.slider("Edad", 18, 92, 30)
    geography = st.sidebar.selectbox("País", ("Francia", "España", "Alemania"))
    
    # Datos Bancarios
    credit_score = st.sidebar.slider("Puntaje de Crédito (Credit Score)", 300, 850, 600)
    tenure = st.sidebar.slider("Años siendo cliente (Tenure)", 0, 10, 3)
    balance = st.sidebar.number_input("Balance en cuenta ($)", 0.0, 250000.0, 60000.0)
    num_of_products = st.sidebar.selectbox("Número de Productos", (1, 2, 3, 4))
    has_cr_card = st.sidebar.checkbox("¿Tiene Tarjeta de Crédito?", value=True)
    is_active_member = st.sidebar.checkbox("¿Es miembro activo?", value=True)
    estimated_salary = st.sidebar.number_input("Salario Estimado ($)", 0.0, 200000.0, 50000.0)

    # Creamos un diccionario con los datos
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

# --- PANTALLA PRINCIPAL ---

# 1. Mostrar los datos ingresados
st.subheader("🔍 Datos del Cliente a Evaluar")
st.dataframe(df, hide_index=True)

# 2. Función de Predicción (MOCKUP / SIMULACIÓN)
# NOTA: Aquí es donde cargarás tu modelo real más adelante con joblib
def predecir_churn_simulado(input_data):
    # Lógica tonta solo para efectos de demostración visual
    # Si es mayor y tiene poco dinero o muchos productos, aumenta el riesgo
    score = 0
    if input_data['Age'][0] > 50: score += 30
    if input_data['IsActiveMember'][0] == 0: score += 20
    if input_data['NumOfProducts'][0] >= 3: score += 40
    if input_data['Balance'][0] == 0: score += 10
    
    # Retornamos probabilidad entre 0 y 1
    probabilidad = min(score + np.random.randint(0, 20), 100) / 100
    return probabilidad

# 3. Botón de Predicción y Resultados
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<br>", unsafe_allow_html=True) # Espacio
    predict_btn = st.button("🚀 Calcular Riesgo de Churn", type="primary")

with col2:
    if predict_btn:
        with st.spinner('Analizando patrones de comportamiento...'):
            time.sleep(1) # Simular tiempo de cómputo
            
            # --- AQUÍ USARÍAS: prediction = model.predict_proba(df) ---
            probabilidad = predecir_churn_simulado(df)
            
            # Mostrar métrica visual
            st.metric(label="Probabilidad de Abandono", value=f"{probabilidad*100:.1f}%")
            
            # Lógica de semáforo
            if probabilidad > 0.5:
                st.error("⚠️ ALTO RIESGO: Es probable que este cliente abandone el banco.")
                st.toast("Alerta: Cliente en riesgo detectado")
            else:
                st.success("✅ BAJO RIESGO: Es probable que el cliente se quede.")

# --- PESTAÑAS EXPLICATIVAS (INTEGRACIÓN NOTEBOOKLM) ---
st.markdown("---")
tab1, tab2 = st.tabs(["📘 Explicación del Modelo", "📊 Importancia de Variables"])

with tab1:
    st.markdown("""
    ### ¿Cómo funciona este modelo?
    *(Aquí pegarás el texto generado por NotebookLM explicando XGBoost o Random Forest)*
    
    Este modelo utiliza un algoritmo de **Gradient Boosting** entrenado con un dataset de 10,000 clientes bancarios.
    Evalúa patrones no lineales entre la edad, el saldo y la actividad del cliente para determinar su fidelidad.
    """)

with tab2:
    st.info("Aquí puedes insertar una imagen estática generada por Matplotlib/SHAP")
    # st.image("assets/feature_importance.png")
    st.markdown("""
    Las variables más influyentes en la decisión del modelo suelen ser:
    1. **Edad**: Los clientes mayores tienden a tener mayor tasa de abandono.
    2. **Número de Productos**: Tener 3 o más productos aumenta drásticamente el riesgo.
    3. **Membresía Activa**: Los miembros inactivos son más propensos a irse.
    """)
