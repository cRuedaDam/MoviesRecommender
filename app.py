import requests
import pandas as pd
import streamlit as st
import os
from src.data.loader import load_data
from src.data.preprocessor import preprocess_data
from src.recommenders.content_based import content_based_recommender, evaluate_content_based
from src.recommenders.collaborative import collaborative_recommender, evaluate_model
from src.recommenders.utils import get_poster_url

# Configuración de la interfaz Streamlit
st.set_page_config(
    page_title="Recomendador de Películas",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados para los elementos visuales
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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    """
    Carga y preprocesa los datos necesarios para el sistema de recomendación.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, scipy.sparse.csr_matrix]:
        metadata, ratings y matriz TF-IDF preprocesada.
    """
    try:
        with st.spinner("Cargando datos, por favor espere..."):
            metadata, ratings, links = load_data(data_folder='dataset')

            if metadata is None or ratings is None or links is None:
                st.error("Error crítico: No se pudieron cargar los archivos de datos.")
                return None, None, None

            metadata, tfidf_matrix = preprocess_data(metadata, links)

            if metadata is None or tfidf_matrix is None:
                st.error("Error crítico: Fallo en el preprocesamiento de datos.")
                return None, None, None
            return metadata, ratings, tfidf_matrix

    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")
        return None, None, None

def load_poster_cache():
    """
    Carga el caché local de URLs de posters si el archivo existe.

    Returns:
        pd.DataFrame: DataFrame con columnas ['tmdbId', 'poster_url']
    """
    cache_file = 'poster_urls_cache.csv'
    
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    return pd.DataFrame(columns=['tmdbId', 'poster_url'])

def save_poster_cache(cache_df):
    """
    Guarda el DataFrame actualizado de caché de posters en disco.

    Args:
        cache_df (pd.DataFrame): DataFrame con tmdbId y poster_url.
    """
    cache_df.to_csv('poster_urls_cache.csv', index=False)

def get_movie_poster_url(movie_info, poster_cache):
    """
    Devuelve la URL del póster de una película, usando caché si está disponible.

    Args:
        movie_info (dict): Información de la película.
        poster_cache (pd.DataFrame): DataFrame de caché de posters.

    Returns:
        str: URL del póster.
    """
    tmdb_id = movie_info['tmdbId']
    movie_title = movie_info.get('title', 'Unknown')
    
    if not tmdb_id or tmdb_id <= 0:
        return "https://via.placeholder.com/300x450?text=Poster+no+disponible"
    
    cached = poster_cache[poster_cache['tmdbId'] == tmdb_id]
    if not cached.empty and pd.notna(cached['poster_url'].iloc[0]):
        return cached['poster_url'].iloc[0]
    
    # Solicitud a la API de TMDB
    poster_url = get_poster_url(tmdb_id, width=342)
    
    # Actualiza y guarda caché
    new_cache_entry = pd.DataFrame({'tmdbId': [tmdb_id], 'poster_url': [poster_url]})
    updated_cache = pd.concat([poster_cache, new_cache_entry]).drop_duplicates(subset='tmdbId', keep='last')
    save_poster_cache(updated_cache)
    
    return poster_url

def display_movie_info(movie_info, poster_cache):
    """
    Muestra la información principal de una película en pantalla.

    Args:
        movie_info (dict): Información de la película.
        poster_cache (pd.DataFrame): DataFrame de caché de posters.
    """
    col1, col2 = st.columns([1, 3])
    poster_url = get_movie_poster_url(movie_info, poster_cache)
    
    with col1:
        try:
            st.image(poster_url, width=300, caption=movie_info['title'], use_container_width=True)
        except Exception:
            st.image("https://via.placeholder.com/300x450?text=Poster+no+disponible",
                     width=300, caption=movie_info['title'], use_container_width=True)
    
    with col2:
        st.markdown(f"""
        **Año:** {int(movie_info['release_year']) if pd.notnull(movie_info.get('release_year')) else 'N/A'}  
        **Géneros:** {', '.join(movie_info.get('genres_list', []))}  
        **Calificación:** {movie_info.get('vote_average', 'N/A')}/10 ⭐  
        **Descripción:**  
        {movie_info.get('overview', 'No hay descripción disponible.')}
        """)

def display_movie_cards(recommendations, metadata, poster_cache, title="Películas Recomendadas", is_collaborative=False):
    """
    Muestra una serie de recomendaciones en formato de tarjetas visuales.

    Args:
        recommendations (list): Lista de tuplas (título, puntuación).
        metadata (pd.DataFrame): DataFrame con la metadata de las películas.
        poster_cache (pd.DataFrame): Caché de pósters.
        title (str): Título de la sección.
        is_collaborative (bool): Si la recomendación es colaborativa, cambia la métrica visualizada.
    """
    st.subheader(title)
    cols = st.columns(5)
    
    for idx, (movie_title, score) in enumerate(recommendations[:10]):
        try:
            movie_info = metadata[metadata['title'] == movie_title].iloc[0]
            poster_url = get_movie_poster_url(movie_info, poster_cache)
            score_display = f"{score:.2f}/10 ⭐" if is_collaborative else f"Similitud: {score:.2f}%"
            with cols[idx % 5]:
                st.markdown(f"""
                <div class="movie-card">
                    <center>
                        <img src="{poster_url}" class="poster"
                             onerror="this.src='https://via.placeholder.com/300x450?text=Poster+no+disponible'; this.onerror=null;">
                        <div class="movie-title" title="{movie_title}">{movie_title}</div>
                        <div class="movie-similarity">{score_display}</div>
                    </center>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error al mostrar {movie_title}: {str(e)}")

# Carga y preprocesamiento de datos al iniciar la aplicación
metadata, ratings, tfidf_matrix = load_and_preprocess_data()
if metadata is None:
    st.stop()

# Limpieza del caché anterior
cache_file = 'poster_urls_cache.csv'
if os.path.exists(cache_file):
    os.remove(cache_file)

poster_cache = load_poster_cache()

# Encabezado de la aplicación
st.title("🎬  Recomendador de Películas")
st.markdown("Descubre películas similares basadas en contenido y preferencias de usuarios.")

# Formulario para seleccionar una película y usuario
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

# Lógica de recomendación al enviar el formulario
if submitted:
    with st.spinner("Generando recomendaciones..."):
        try:
            movie_info = metadata[metadata['title'] == movie_title].iloc[0]
            
            st.subheader(f"🎥    {movie_title}", divider="blue")
            display_movie_info(movie_info, poster_cache)
            
            st.subheader("🍿    Recomendaciones basadas en contenido", divider="blue")
            content_recs = content_based_recommender(
                movie_title, metadata, n_recommendations=10, tfidf_matrix=tfidf_matrix
            )
            display_movie_cards(content_recs, metadata, poster_cache)

            with st.expander("📊  Ver detalles técnicos (contenido)"):
                st.dataframe(
                    pd.DataFrame(content_recs, columns=['Título', 'Similitud (%)']),
                    use_container_width=True,
                    hide_index=True
                )

            st.subheader("👥    También les gustó a usuarios similares", divider="blue")
            collab_recs = collaborative_recommender(
                user_id, ratings, metadata, n_recommendations=10, input_title=movie_title
            )
            display_movie_cards(collab_recs, metadata, poster_cache, title="", is_collaborative=True)

            with st.expander("📊  Ver detalles técnicos (colaborativo)"):
                st.dataframe(
                    pd.DataFrame(collab_recs, columns=['Título', 'Puntuación media']),
                    use_container_width=True,
                    hide_index=True
                )

            #Implementacion de pruebas
            #metrics = evaluate_model(collaborative_recommender, ratings, metadata, k=10, n_users=100)
            #metrics = evaluate_content_based(metadata, tfidf_matrix, content_based_recommender, n_recommendations=10, test_samples=50, k=10)
            #print(metrics)


        except Exception as e:
            st.error(f"Error al generar recomendaciones: {str(e)}")
            st.error("Por favor intenta con otra película o reinicia la aplicación.")


