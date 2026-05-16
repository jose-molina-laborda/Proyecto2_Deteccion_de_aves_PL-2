import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# --- 1. CONFIGURACION DE LA PAGINA ---
st.set_page_config(
    page_title="IA Clasificador de Aves por Atributos",
    layout="centered"
)

st.title("Clasificador de Aves por Atributos Descriptivos")
st.markdown("""
Esta aplicación utiliza el modelo de Bosque Aleatorio (Random Forest) entrenado 
para identificar especies de aves basándose en la presencia de sus atributos 
descriptivos, no en imágenes.
""")

# --- 2. CARGA DE RECURSOS (CACHED) ---
@st.cache_resource
def load_resources():

    # Cargar nombres de clases
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

    # Cargar modelo
    try:
        model = joblib.load("modelo_aves_rf.pkl")
    except Exception as e:
        st.error(f"Error al cargar modelo_aves_rf.pkl: {e}")
        model = None

    # Cargar nombres de atributos en español
    try:
        with open("atributos_es.json", "r", encoding="utf-8") as f:
            atributos_es = json.load(f)
    except Exception as e:
        st.error(f"Error al cargar atributos_es.json: {e}")
        atributos_es = {}

    return model, class_names, atributos_es


model, class_names, atributos_es = load_resources()

# --- 3. INTERFAZ DE USUARIO ---
if model is not None and class_names is not None:

    st.subheader("Selección de atributos")
    st.write("Selecciona todos los atributos (del 1 al 312) presentes en el ave:")

    opciones_atributos = list(range(1, 313))

    # Función para mostrar nombre en español
    def mostrar_nombre(id_attr):
        return atributos_es.get(str(id_attr), f"Atributo {id_attr}")

    atributos_seleccionados = st.multiselect(
        "Atributos presentes:",
        options=opciones_atributos,
        format_func=mostrar_nombre
    )

    if st.button("Identificar Especie"):

        features = np.zeros(312)

        for attr_id in atributos_seleccionados:
            features[attr_id - 1] = 1

        columnas = list(range(1, 313))
        input_df = pd.DataFrame([features], columns=columnas)

        with st.spinner("Analizando la combinación de atributos..."):

            clase_predicha = model.predict(input_df)[0]
            probabilidades = model.predict_proba(input_df)[0]
            confianza = np.max(probabilidades)

            clases_modelo = list(model.classes_)
            indice_clase = clases_modelo.index(clase_predicha)
            nombre_especie = class_names[indice_clase]

            st.success(f"Resultado: **{nombre_especie}**")
            st.write(f"Confianza del modelo: {confianza * 100:.2f}%")
            st.progress(float(confianza))

            if confianza < 0.40:
                st.warning("La confianza es baja. Intenta seleccionar más atributos.")
else:
    st.warning("Asegúrate de haber subido 'classes.txt', 'modelo_aves_rf.pkl' y 'atributos_es.json'.")
