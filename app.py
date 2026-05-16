import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. CONFIGURACION DE LA PAGINA ---
st.set_page_config(
    page_title="IA Clasificador de Aves por Atributos",
    layout="centered"
)

st.title("Clasificador de Aves por Atributos Descriptivos")
st.markdown("""
Esta aplicacion utiliza el modelo de Bosque Aleatorio (Random Forest) entrenado 
para identificar especies de aves basandose en la presencia de sus atributos 
descriptivos, no en imagenes.
""")

# --- 2. CARGA DE RECURSOS (CACHED) ---
@st.cache_resource
def load_resources():
    # Cargar nombres de las clases desde classes.txt
    try:
        with open("classes.txt", "r") as f:
            class_names = []
            for line in f:
                if line.strip():
                    name_part = line.strip().split(' ')[1].split('.')[1]
                    clean_name = name_part.replace('_', ' ')
                    class_names.append(clean_name)
    except Exception as e:
        st.error(f"Error al cargar classes.txt: {e}")
        class_names = None

    # Cargar el modelo Random Forest (.pkl)
    try:
        model = joblib.load("modelo_aves_rf.pkl")
    except Exception as e:
        st.error(f"Error al cargar el archivo modelo_aves_rf.pkl: {e}")
        model = None

    return model, class_names

model, class_names = load_resources()

# --- 3. INTERFAZ DE USUARIO ---
if model is not None and class_names is not None:
    st.subheader("Seleccion de atributos")
    st.write("Selecciona todos los atributos (del 1 al 312) que esten presentes en el ave que deseas identificar:")

    # Lista con los 312 identificadores de atributos
    opciones_atributos = list(range(1, 313))
    
    # Selector multiple para que el usuario escoja que atributos marcar como "1"
    atributos_seleccionados = st.multiselect(
        "Atributos presentes:",
        options=opciones_atributos,
        format_func=lambda x: f"Atributo {x}"
    )

    if st.button("Identificar Especie"):
        # Inicializar vector de 312 ceros
        features = np.zeros(312)
        
        # Poner en 1 los atributos que el usuario haya seleccionado (indice ajustado -1)
        for attr_id in atributos_seleccionados:
            features[attr_id - 1] = 1 
            
        # Crear DataFrame para la prediccion, ya que entrenamos con un DataFrame con columnas numéricas
        columnas = list(range(1, 313))
        input_df = pd.DataFrame([features], columns=columnas)
        
        with st.spinner("Analizando la combinacion de atributos..."):
            # Realizar prediccion
            clase_predicha = model.predict(input_df)[0]
            
            # Obtener probabilidad / confianza
            probabilidades = model.predict_proba(input_df)[0]
            confianza = np.max(probabilidades)
            
            # Mapear la clase al nombre. 
            # model.classes_ tiene las clases en el orden exacto que Random Forest las aprendio
            clases_modelo = list(model.classes_)
            indice_clase = clases_modelo.index(clase_predicha)
            nombre_especie = class_names[indice_clase]
            
            st.success(f"Resultado: **{nombre_especie}**")
            st.write(f"Confianza del modelo: {confianza * 100:.2f}%")
            st.progress(float(confianza))
            
            if confianza < 0.40:
                st.warning("La confianza es algo baja. Intenta afinar mas tu seleccion de atributos.")
else:
    st.warning("Asegurate de haber subido los archivos 'classes.txt' y 'modelo_aves_rf.pkl' al repositorio.")
