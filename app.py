import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Configuración de la página (Modo ancho y título en la pestaña del navegador)
st.set_page_config(
    page_title="Spotify SVD Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado para darle un toque "Spotify Black & Green"
# CORREGIDO: unsafe_allow_html
st.markdown("""
    <style>
    .main { background-color: #121212; color: #FFFFFF; }
    .stButton>button { background-color: #1DB954; color: white; border-radius: 20px; }
    .stProgress > div > div > div > div { background-color: #1DB954; }
    div[data-testid="stExpander"] { background-color: #191919; border: 1px solid #282828; }
    </style>
    """, unsafe_allow_html=True)

# Función optimizada para cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv("datos_spotify.csv")
    df.set_index('Usuario', inplace=True)
    return df

try:
    df_matriz = cargar_datos()
    usuarios = df_matriz.index.tolist()
    canciones = df_matriz.columns.tolist()

    # ==========================================
    # BARRA LATERAL (SIDEBAR) - CONFIGURACIÓN
    # ==========================================
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg", width=150)
        st.markdown("## **Panel de Control**")
        st.write("Configura los parámetros del algoritmo de Álgebra Lineal.")
        
        st.markdown("---")
        usuario_elegido = st.selectbox("👤 **Selecciona un Usuario:**", usuarios)
        num_recomendaciones = st.slider("💿 **Número de sugerencias:**", min_value=1, max_value=4, value=2)
        
        st.markdown("---")
        st.markdown("### **Ajustes del Algoritmo**")
        k_componentes = st.slider("🧮 Componentes Latentes (SVD K):", min_value=2, max_value=5, value=3)

    # ==========================================
    # CUERPO PRINCIPAL - DISEÑO POR PESTAÑAS
    # ==========================================
    st.title("🎵 Spotify Recommendation Engine")
    st.write("Prototipo inteligente basado en **Descomposición en Valores Singulares (SVD)**.")
    
    # Creamos dos pestañas para organizar el contenido
    tab1, tab2 = st.tabs(["🚀 Descubrir Música", "📊 Consola del Profesor (Álgebra Lineal)"])

    # ------------------------------------------
    # PESTAÑA 1: INTERFAZ DE USUARIO
    # ------------------------------------------
    with tab1:
        # Procesamiento matemático interno
        matriz_canciones = df_matriz.T 
        svd = TruncatedSVD(n_components=k_componentes, random_state=42)
        matriz_reducida = svd.fit_transform(matriz_canciones)
        similitud_conceptos = np.dot(matriz_reducida, matriz_reducida.T)
        df_similitud = pd.DataFrame(similitud_conceptos, index=canciones, columns=canciones)

        # Historial del usuario elegido
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
                puntuaciones = df_similitud[canciones_escuchadas].sum(axis=1)
                recomendaciones_ordenadas = puntuaciones.sort_values(ascending=False)
                nuevas_recomendaciones = recomendaciones_ordenadas.drop(canciones_escuchadas)
                top_final = nuevas_recomendaciones.head(num_recomendaciones)

                st.caption("Analizando cercanía de vectores en el espacio latente...")
                
                # Desplegar las recomendaciones como tarjetas limpias
                # CORREGIDO: unsafe_allow_html en la ejecución de las tarjetas
                for idx, (cancion, score) in enumerate(top_final.items(), 1):
                    with st.container():
                        st.markdown(f"""
                        <div style='background-color: #242424; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #1DB954;'>
                            <span style='color: #1DB954; font-weight: bold; font-size: 18px;'>#{idx} {cancion}</span><br>
                            <span style='color: #B3B3B3; font-size: 14px;'>Similitud Geométrica: <b>{score:.2f}</b></span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Este usuario no registra suficiente actividad para calcular vectores.")

    # ------------------------------------------
    # PESTAÑA 2: SUSTENTO MATEMÁTICO (PARA EXPOSICIÓN)
    # ------------------------------------------
    with tab2:
        st.markdown("### 🧠 Caja Negra del Algoritmo SVD")
        st.write("Usa esta pestaña durante la exposición para explicar el procesamiento de matrices al profesor.")
        
        st.markdown("#### **1. Matriz de Entrada Original ($A$)**")
        st.dataframe(df_matriz, use_container_width=True)
        
        st.markdown("#### **2. Reducción de Dimensionalidad ($V^T$ Truncada)**")
        st.write(f"Tamaño reducido a: **{matriz_reducida.shape[0]} canciones y {matriz_reducida.shape[1]} componentes latentes**.")
        
        df_reducido_ver = pd.DataFrame(matriz_reducida, index=canciones, columns=[f"Concepto {i+1}" for i in range(k_componentes)])
        st.dataframe(df_reducido_ver, use_container_width=True)
        
        st.markdown("#### **3. Matriz de Similitud (Producto Punto: $V \\cdot V^T$)**")
        st.write("Resultado de multiplicar los vectores reducidos para medir qué tan alineados están en el espacio.")
        st.dataframe(df_similitud, use_container_width=True)

except FileNotFoundError:
    st.error("Error: No se encuentra el archivo 'datos_spotify.csv'. Asegúrate de que esté en la misma carpeta.")