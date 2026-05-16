import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

# ==============================================================================
# PARTE 1: Carga del dataset y preprocesado de los datos
# ==============================================================================
print("1. Cargando y preprocesando datos...")

# 1.1 Cargar atributos
df_atributos = pd.read_csv('image_attribute_labels.txt', sep='\s+', header=None,
                           usecols=[0, 1, 2], names=['imgid', 'attribute_id', 'is_present'])

# 1.2 Reorganizar los datos (1 fila por imagen, 312 columnas con valores 1/0)
df_pivot = df_atributos.pivot(index='imgid', columns='attribute_id', values='is_present')

# 1.3 Cargar las clases verdaderas
df_clases = pd.read_csv('image_class_labels.txt', sep=' ', header=None, names=['imgid', 'class_id'])
df_clases = df_clases.set_index('imgid')

# 1.4 Juntar atributos y etiquetas (join)
df_final = df_pivot.join(df_clases, how='inner')

# 1.5 Barajar los datos para evitar patrones de orden
df_final = df_final.sample(frac=1, random_state=42)

# 1.6 Dividir en variables (X) y etiquetas (y), y luego en train y test
X = df_final.drop('class_id', axis=1)
y = df_final['class_id']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================================================================
# PARTE 2: Entrenamiento Anti-Overfitting Extremo y Guardado
# ==============================================================================
# ==============================================================================
# PARTE 2: Entrenamiento Equilibrado (El punto dulce)
# ==============================================================================
print("2. Entrenando modelo en el punto intermedio...")

rf_optimizado = RandomForestClassifier(
    n_estimators=300,           # Aumentamos los árboles para mayor estabilidad
    max_features='sqrt',
    max_depth=15,               # Le devolvemos cierta libertad para hacer preguntas profundas
    min_samples_split=12,        # Freno intermedio: exige 8 aves para ramificar
    min_samples_leaf=4,         # Freno intermedio: las hojas finales deben tener al menos 2 aves
    class_weight='balanced',
    random_state=42,
    n_jobs=1
)

# Entrenar el modelo
rf_optimizado.fit(X_train, y_train)

# Evaluar en Entrenamiento y Test
precision_train = rf_optimizado.score(X_train, y_train)
predicciones_opt = rf_optimizado.predict(X_test)
precision_test = rf_optimizado.score(X_test, y_test)

print(f"   -> Precisión en Entrenamiento (Estudio): {precision_train:.4f} ({precision_train*100:.2f}%)")
print(f"   -> Precisión en Test (Examen): {precision_test:.4f} ({precision_test*100:.2f}%)")

# Guardar el modelo para Streamlit y GitHub (CON COMPRESION)
nombre_archivo = 'modelo_aves_rf.pkl'

# El parametro compress=3 reduce el tamaño del archivo entre un 70% y un 90%
joblib.dump(rf_optimizado, nombre_archivo, compress=9)

print(f"   -> Modelo guardado exitosamente como '{nombre_archivo}'")
# ==============================================================================
# PARTE 3: Matriz de confusion
# ==============================================================================
print("3. Generando Matriz de Confusion...")

# Cargar nombres de las clases desde classes.txt
with open('classes.txt', 'r') as f:
    nombres_clases = [line.strip().split(' ')[1] for line in f]

# Crear matriz de confusion usando las predicciones del modelo optimizado
matriz = confusion_matrix(y_test, predicciones_opt)

# Dibujar la matriz usando el ejemplo de sklearn
fig, ax = plt.subplots(figsize=(15, 15))
disp = ConfusionMatrixDisplay(confusion_matrix=matriz)
disp.plot(ax=ax, cmap='Blues', include_values=False, colorbar=True)
plt.title('Matriz de Confusion - 200 Especies')
plt.show()

# ==============================================================================
# PARTE 4: Bucle de hiperparametros y Grafico Scatter
# ==============================================================================
print("4. Iterando combinaciones de hiperparametros (Rutina manual)...")

lista_estimadores = [50, 100, 150]
lista_max_features = [10, 50, 100]

resultados_x_features = []
resultados_y_estimadores = []
resultados_z_precision = []

# Rutina manual para probar combinaciones
for n_est in lista_estimadores:
    for m_feat in lista_max_features:
        rf_temp = RandomForestClassifier(max_features=m_feat, n_estimators=n_est, random_state=0, n_jobs=1)
        rf_temp.fit(X_train, y_train)
        acc = rf_temp.score(X_test, y_test)

        print(f"   -> n_estimators: {n_est} | max_features: {m_feat} | Accuracy: {acc:.4f}")

        # Guardar para el grafico
        resultados_x_features.append(m_feat)
        resultados_y_estimadores.append(n_est)
        resultados_z_precision.append(acc)

print("5. Mostrando evolucion con grafico scatter...")

# Mostrar mediante grafico de tipo scatter
plt.figure(figsize=(8, 6))
scatter = plt.scatter(resultados_x_features, resultados_y_estimadores,
                      c=resultados_z_precision, cmap='viridis', s=200, edgecolor='k')

plt.colorbar(scatter, label='Accuracy (Precision)')
plt.xlabel('Numero maximo de atributos (max_features)')
plt.ylabel('Numero de arboles (n_estimators)')
plt.title('Evolucion de la Precision del Random Forest')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(lista_max_features)
plt.yticks(lista_estimadores)
plt.show()

print("Proyecto completado. Ya puedes descargar el archivo .pkl desde el panel izquierdo.")
