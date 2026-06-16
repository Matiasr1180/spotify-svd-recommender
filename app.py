import streamlit as st
import pandas as pd
import numpy as np

# Configuración de la interfaz y ventana
st.set_page_config(
    page_title="Spotify SVD Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado para la identidad de Spotify
st.markdown("""
    <style>
    .main { background-color: #121212; color: #FFFFFF; }
    .stButton>button { background-color: #1DB954; color: white; border-radius: 20px; }
    div[data-testid="stExpander"] { background-color: #191919; border: 1px solid #282828; }
    </style>
    """, unsafe_allow_html=True)

# Función modular para la carga de datos optimizada en caché
@st.cache_data
def cargar_datos():
    df = pd.read_csv("datos_spotify.csv")
    df.set_index('Usuario', inplace=True)
    return df

try:
    df_matriz = cargar_datos()
    usuarios = df_matriz.index.tolist()
    canciones = df_matriz.columns.tolist()

    # Componentes de entrada interactivos en la barra lateral (Sidebar)
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg", width=150)
        st.markdown("## **Panel de Control**")
        st.write("Configura los parámetros del algoritmo de Álgebra Lineal.")
        st.markdown("---")
        usuario_elegido = st.selectbox("👤 **Selecciona un Usuario:**", usuarios)
        num_recomendaciones = st.slider("💿 **Número de sugerencias:**", min_value=1, max_value=4, value=2)
        st.markdown("---")
        st.markdown("### **Ajustes del Algoritmo**")
        # En SVD normal, "K" actúa seleccionando las columnas del resultado de la factorización exacta
        k_componentes = st.slider("🧮 Canciones referenciales (K):", min_value=2, max_value=5, value=3)

    # Organización de la aplicación mediante Pestañas (Tabs)
    st.title("🎵 Spotify Recommendation Engine")
    st.write("Prototipo de mi proyecto final de ALC **Descomposición en Valores Singulares (SVD Normal)**.")
    
    # Creamos dos pestañas para organizar el contenido
    tab1, tab2 = st.tabs(["🚀 Descubrir Música", "📊 Base de Datos"])

    # PESTAÑA 1: EXPERIENCIA DEL USUARIO
    with tab1:
        # Transposición matemática: cambiar filas por columnas (Álgebra Lineal)
        matriz_canciones = df_matriz.T 
        
        # =========================================================================
        # NUEVO CORE DE ÁLGEBRA LINEAL: SVD ESTÁNDAR CON NUMPY
        # =========================================================================
        # np.linalg.svd factoriza de forma exacta: Matriz = U * Sigma * Vt
        # full_matrices=False nos da la descomposición económica (m x k)
        U, S, Vt = np.linalg.svd(matriz_canciones.values, full_matrices=False)
        
        # Proyectamos las canciones en el espacio latente quedándonos con las primeras 'K' columnas de U
        matriz_reducida = U[:, :k_componentes]
        
        # Producto punto para calcular similitudes geométricas angulares
        similitud_conceptos = np.dot(matriz_reducida, matriz_reducida.T)
        df_similitud = pd.DataFrame(similitud_conceptos, index=canciones, columns=canciones)
        # =========================================================================

        # Extracción del historial de escucha del cliente
        historial = df_matriz.loc[usuario_elegido]
        canciones_escuchadas = historial[historial == 1].index.tolist()

        col_izq, col_der = st.columns([1, 2])

        with col_izq:
            st.markdown(f"### ✨ Perfil de **{usuario_elegido}**")
            st.write("**Canciones actuales en su Playlist:**")
            for cancion in canciones_escuchadas:
                st.markdown(f"- 🎧 {cancion}")

        with col_der:
            st.markdown("### 🎯 Recomendaciones Generadas")
            if canciones_escuchadas:
                # Suma algebraica de scores vectoriales
                puntuaciones = df_similitud[canciones_escuchadas].sum(axis=1)
                recomendaciones_ordenadas = puntuaciones.sort_values(ascending=False)
                
                # Filtrado condicional para evitar redundancia
                nuevas_recomendaciones = recomendaciones_ordenadas.drop(canciones_escuchadas)
                top_final = nuevas_recomendaciones.head(num_recomendaciones)

                st.caption("Recomendaciones generadas en base a su playlist:")
                
                # Desplegar resultados simulando tarjetas en la UI
                for idx, (cancion, score) in enumerate(top_final.items(), 1):
                    with st.container():
                        st.markdown(f"""
                        <div style='background-color: #242424; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #1DB954;'>
                            <span style='color: #1DB954; font-weight: bold; font-size: 18px;'>#{idx} {cancion}</span><br>
                            <span style='color: #B3B3B3; font-size: 14px;'>Similitud: <b>{score:.2f}</b></span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Este usuario no registra suficiente actividad para calcular vectores.")

    # PESTAÑA 2: CONSOLA AUDITORÍA DEL DOCENTE (MARCO MATEMÁTICO)
    with tab2:
        st.markdown("### 🧠 Datos del Algoritmo")
        st.write("Explicación del funcionamiento interno del algoritmo.")
        
        st.markdown("#### **1. Matriz de Entrada Original ($A$)**")
        st.dataframe(df_matriz, use_container_width=True)
        
        st.markdown("#### **2. Reducción mediante SVD Estándar ($U$ de NumPy)**")
        st.write(f"Tamaño extraído de la matriz ortogonal $U$: **{matriz_reducida.shape[0]} canciones y {matriz_reducida.shape[1]} componentes seleccionados**.")
        
        df_reducido_ver = pd.DataFrame(matriz_reducida, index=canciones, columns=[f"Genero {i+1}" for i in range(k_componentes)])
        st.dataframe(df_reducido_ver, use_container_width=True)
        
        st.markdown("#### **3. Matriz de Similitud (Producto Punto: $U_k \\cdot U_k^T$)**")
        st.write("Resultado de multiplicar los vectores reducidos para medir qué tan alineados están en el espacio.")
        st.dataframe(df_similitud, use_container_width=True)

except FileNotFoundError:
    st.error("Error crítico: El archivo de base de datos 'datos_spotify.csv' no fue encontrado en el directorio raíz.")