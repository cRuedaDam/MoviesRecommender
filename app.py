import requests
import pandas as pd
import streamlit as st
from src.data.loader import load_data
from src.data.preprocessor import preprocess_data
from src.recommenders.content_based import content_based_recommender
from src.recommenders.collaborative import collaborative_recommender
from src.recommenders.utils import get_poster_url  # Función auxiliar para obtener URL del póster

# Configura la interfaz de usuario de Streamlit
st.set_page_config(
    page_title="Recomendador de Películas",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de estilos personalizados en la interfaz Streamlit
st.markdown("""
<style>
.poster {
    height: 300px;
    border-radius: 10px;
    transition: transform 0.3s;
    object-fit: cover;
    background: #2a2a2a;
}
.poster:hover {
    transform: scale(1.05);
}
.movie-card {
    padding: 15px;
    border-radius: 10px;
    background: #1a1a1a;
    transition: all 0.3s;
    height: 420px;
    overflow: hidden;
    margin-bottom: 20px;
}
.movie-card:hover {
    background: #252525;
}
.movie-title {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin: 10px 0 5px 0;
    font-weight: bold;
    font-size: 14px;
    color: #f0f0f0;
}
.movie-similarity {
    font-size: 13px;
    color: #f0f0f0;
    margin: 0;
}
.selectbox-container {
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    """
    Carga los datos desde los archivos fuente y aplica el preprocesamiento necesario.

    Retorna:
        metadata (DataFrame): información de las películas.
        ratings (DataFrame): valoraciones de los usuarios.
        tfidf_matrix (sparse matrix): matriz TF-IDF para similitud de contenido.
    """
    try:
        with st.spinner("Cargando datos, por favor espere..."):
            # Carga los datos desde los archivos CSV en la carpeta 'dataset'
            metadata, ratings, links = load_data(data_folder='dataset')
            
            # Validación de carga exitosa
            if metadata is None or ratings is None or links is None:
                st.error("Error crítico: No se pudieron cargar los archivos de datos.")
                return None, None, None
            
            # Preprocesamiento y vectorización de metadatos
            metadata, tfidf_matrix = preprocess_data(metadata, links)
            
            # Validación de preprocesamiento exitoso
            if metadata is None or tfidf_matrix is None:
                st.error("Error crítico: Fallo en el preprocesamiento de datos.")
                return None, None, None
            
            # Asegura la existencia de una columna para validar disponibilidad del póster
            if 'has_valid_poster' not in metadata.columns:
                metadata['has_valid_poster'] = False
            
            return metadata, ratings, tfidf_matrix
    
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")
        return None, None, None

def display_movie_info(movie_info):
    """
    Muestra detalles de una película seleccionada.

    Argumentos:
        movie_info (Series): contiene la información de la película, como título, póster, año, géneros y descripción.
    """
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(
            movie_info['poster_url'],
            width=300,
            caption=movie_info['title'],
            use_container_width=True
        )
    with col2:
        st.markdown(f"""
        **Año:** {int(movie_info['release_year']) if pd.notnull(movie_info.get('release_year')) else 'N/A'}  
        **Géneros:** {', '.join(movie_info.get('genres_list', []))}  
        **Calificación:** {movie_info.get('vote_average', 'N/A')}/10 ⭐  
        **Descripción:**  
        {movie_info.get('overview', 'No hay descripción disponible.')}
        """)

def display_movie_cards(recommendations, metadata, title="Películas Recomendadas", is_collaborative=False):
    """
    Presenta una lista de películas recomendadas en formato de tarjetas visuales.

    Argumentos:
        recommendations (list): lista de tuplas (título, puntuación).
        metadata (DataFrame): datos de las películas.
        title (str): título del bloque de recomendaciones.
        is_collaborative (bool): define si el origen es colaborativo o basado en contenido.
    """
    st.subheader(title)
    cols = st.columns(5)
    
    for idx, (movie_title, score) in enumerate(recommendations[:10]):
        try:
            movie_info = metadata[metadata['title'] == movie_title].iloc[0]
            score_display = f"{score:.2f}/10 ⭐" if is_collaborative else f"Similitud: {score:.2f}%"
            with cols[idx % 5]:
                st.markdown(f"""
                <div class="movie-card">
                    <center>
                        <img src="{movie_info['poster_url']}" class="poster"
                             onerror="this.src='https://via.placeholder.com/300x450?text=Poster+no+disponible'; this.onerror=null;">
                        <div class="movie-title" title="{movie_title}">{movie_title}</div>
                        <div class="movie-similarity">{score_display}</div>
                    </center>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error al mostrar {movie_title}: {str(e)}")

# Carga inicial de datos
metadata, ratings, tfidf_matrix = load_and_preprocess_data()

# Interrumpe la ejecución si no se pudieron cargar los datos
if metadata is None:
    st.stop()

# Encabezado de la aplicación
st.title("🎬  Recomendador de Películas")
st.markdown("Descubre películas similares basadas en contenido y preferencias de usuarios.")

# Formulario para seleccionar película e ingresar ID de usuario
with st.form(key="recommendation_form"):
    st.subheader("Buscar recomendaciones")
    col1, col2 = st.columns([3, 1])
    with col1:
        movie_title = st.selectbox(
            "Selecciona una película",
            options=sorted(metadata['title'].tolist()),
            index=0,
            key="movie_select"
        )
    with col2:
        user_id = st.number_input(
            "ID de usuario",
            min_value=1,
            value=1,
            step=1,
            help="Ingresa un número entre 1 y 138493"
        )
    submitted = st.form_submit_button("Obtener recomendaciones")

# Procesamiento cuando el usuario envía el formulario
if submitted:
    with st.spinner("Generando recomendaciones..."):
        try:
            # Información de la película seleccionada
            movie_info = metadata[metadata['title'] == movie_title].iloc[0]
            
            st.subheader(f"🎥  {movie_title}", divider="blue")
            display_movie_info(movie_info)
            
            # Recomendaciones basadas en contenido
            st.subheader("🍿  Recomendaciones basadas en contenido", divider="blue")
            content_recs = content_based_recommender(
                movie_title, 
                metadata, 
                n_recommendations=10, 
                tfidf_matrix=tfidf_matrix
            )
            display_movie_cards(content_recs, metadata)

            with st.expander("📊 Ver detalles técnicos (contenido)"):
                st.dataframe(
                    pd.DataFrame(content_recs, columns=['Título', 'Similitud (%)']),
                    use_container_width=True,
                    hide_index=True
                )

            # Recomendaciones basadas en comportamiento de usuarios
            st.subheader("👥  También les gustó a usuarios similares", divider="blue")
            collab_recs = collaborative_recommender(
                user_id, 
                ratings, 
                metadata, 
                n_recommendations=10, 
                input_title=movie_title
            )
            display_movie_cards(collab_recs, metadata, title="", is_collaborative=True)

            with st.expander("📊 Ver detalles técnicos (colaborativo)"):
                st.dataframe(
                    pd.DataFrame(collab_recs, columns=['Título', 'Puntuación media']),
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as e:
            st.error(f"Error al generar recomendaciones: {str(e)}")
            st.error("Por favor intenta con otra película o reinicia la aplicación.")
