import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import requests

# --- 1. CONFIGURACION DE LA PAGINA ---
st.set_page_config(
    page_title="IA Clasificador de Aves por Atributos",
    layout="centered"
)

st.title("Clasificador de Aves por Atributos Descriptivos")
st.markdown("""
Esta aplicación utiliza el modelo de Bosque Aleatorio (Random Forest) entrenado 
para identificar especies de aves basándose en la presencia de sus atributos 
descriptivos. Además, recupera una imagen de la web para ilustrar el resultado.
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

    # Cargar atributos estructurados por categorías
    try:
        with open("atributos_categorias.json", "r", encoding="utf-8") as f:
            atributos_cat = json.load(f)
    except Exception as e:
        st.error("Por favor, crea el archivo 'atributos_categorias.json' con las secciones.")
        atributos_cat = {}

    return model, class_names, atributos_cat

model, class_names, atributos_cat = load_resources()

# --- FUNCIÓN PARA BUSCAR IMAGEN EN WIKIPEDIA ---
def obtener_imagen_wikipedia(nombre_especie):
    try:
        # Usamos la API de Wikipedia en inglés ya que los nombres originales están en ese idioma
        url = f"https://en.wikipedia.org/w/api.php?action=query&titles={nombre_especie}&prop=pageimages&format=json&pithumbsize=500"
        response = requests.get(url).json()
        pages = response.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if "thumbnail" in page_data:
                return page_data["thumbnail"]["source"]
    except Exception:
        return None
    return None

# --- 3. INTERFAZ DE USUARIO ---
if model is not None and class_names is not None and atributos_cat:
    
    st.subheader("Selección de características")
    st.write("Selecciona una opción por cada categoría. Si no estás seguro de una característica, déjala en 'Desconocido'.")
    
    ids_seleccionados = []
    
    # Organizamos la interfaz en 2 columnas para que sea más fácil de leer
    cols = st.columns(2)
    
    for i, (categoria, opciones) in enumerate(atributos_cat.items()):
        col = cols[i % 2]
        with col:
            # Se añade la opción "vacía" por defecto para no obligar a seleccionar todo
            lista_opciones = [(None, "Desconocido / No aplica")]
            for attr_id, attr_name in opciones.items():
                lista_opciones.append((int(attr_id), attr_name))
            
            # st.selectbox garantiza que el usuario SOLO pueda elegir un tipo de pico, ala, etc.
            seleccion = st.selectbox(
                categoria, 
                options=lista_opciones, 
                format_func=lambda x: x[1],
                key=categoria
            )
            
            # Guardamos el ID solo si ha elegido un rasgo real
            if seleccion[0] is not None:
                ids_seleccionados.append(seleccion[0])
    
    st.write("---")
    
    # Botón principal
    if st.button("Identificar Especie", use_container_width=True):
        
        # Inicializamos los 312 atributos en 0
        features = np.zeros(312)
        
        for attr_id in ids_seleccionados:
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
                st.warning("La confianza es baja. Intenta afinar más tu selección de atributos.")
                
            # --- SECCIÓN DE LA FOTOGRAFÍA ---
            st.write("### Aspecto de la especie")
            url_imagen = obtener_imagen_wikipedia(nombre_especie)
            
            if url_imagen:
                st.image(url_imagen, caption=f"Imagen obtenida automáticamente de Wikipedia", use_container_width=True)
            else:
                st.info("No se encontró una foto pública en Wikipedia para esta especie concreta.")
                # Enlace interactivo y dinámico a Google Imágenes
                url_google = f"https://www.google.com/search?tbm=isch&q={nombre_especie.replace(' ', '+')}"
                st.markdown(f"[👉 **Haz clic aquí para buscar el {nombre_especie} en Google Imágenes**]({url_google})", unsafe_allow_html=True)

else:
    st.warning("Asegúrate de haber subido 'classes.txt', 'modelo_aves_rf.pkl' y el nuevo archivo de secciones 'atributos_categorias.json'.")
