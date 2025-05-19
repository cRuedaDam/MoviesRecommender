import streamlit as st
import pandas as pd
import numpy as np
from recommender import content_based_recommender, collaborative_recommender, load_and_preprocess_data

# Configuración de la página
st.set_page_config(page_title="Recomendador de Películas", page_icon="🎬", layout="wide")

# Título y descripción
st.title("🎬 Recomendador de Películas")
st.markdown("Selecciona una película y un ID de usuario para obtener recomendaciones personalizadas basadas en contenido y filtrado colaborativo.")

# Cargar datos usando la función de recommender.py
@st.cache_data
def load_data():
    st.write("Cargando datos...")
    metadata, ratings, tfidf_matrix = load_and_preprocess_data(data_folder='dataset')
    if metadata is None or ratings is None or tfidf_matrix is None:
        st.error("No se pudieron cargar los datos. Verifica que los archivos estén en la carpeta 'dataset/'.")
        return None, None, None
    st.write("Datos cargados.")
    return metadata, ratings, tfidf_matrix

metadata, ratings, tfidf_matrix = load_data()

# Verificar si los datos se cargaron correctamente
if metadata is None:
    st.stop()

# Formulario
with st.form(key="recommendation_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        movie_title = st.selectbox("Selecciona una película", options=[""] + sorted(metadata['title'].tolist()), index=0)
    with col2:
        user_id = st.number_input("ID de usuario", min_value=1, value=1, step=1)
    submit_button = st.form_submit_button("Obtener Recomendaciones")

# Procesar recomendaciones
if submit_button and movie_title:
    if movie_title not in metadata['title'].values:
        st.error("Película no encontrada. Por favor, selecciona una película válida.")
    else:
        with st.spinner("Generando recomendaciones..."):
            try:
                # Obtener movieId
                movie_id = metadata[metadata['title'] == movie_title]['movieId'].iloc[0]

                # Obtener recomendaciones
                content_recs = content_based_recommender(movie_title, metadata, n_recommendations=10, tfidf_matrix=tfidf_matrix)
                collab_recs = collaborative_recommender(user_id, ratings, metadata, n_recommendations=10, input_title=movie_title, tfidf_matrix=tfidf_matrix)

                # Mostrar información de la película seleccionada
                movie_info = metadata[metadata['title'] == movie_title][['title', 'release_year', 'genres_list']].iloc[0]
                st.subheader(f"Película seleccionada: {movie_title}")
                st.write(f"Año: {movie_info['release_year']}")
                st.write(f"Géneros:47 {', '.join(movie_info['genres_list'])}")

                # Mostrar recomendaciones basadas en contenido
                st.subheader("Recomendaciones basadas en contenido")
                content_df = pd.DataFrame(content_recs, columns=['Título', 'Similitud (%)'])
                content_df['Similitud (%)'] = content_df['Similitud (%)'].round(2)
                st.dataframe(
                    content_df,
                    use_container_width=True,
                    column_config={
                        "Título": st.column_config.TextColumn("Título"),
                        "Similitud (%)": st.column_config.NumberColumn("Similitud (%)", format="%.2f")
                    }
                )

                # Mostrar recomendaciones colaborativas
                st.subheader("Recomendaciones colaborativas")
                collab_df = pd.DataFrame(collab_recs, columns=['Título', 'Relevancia (%)'])
                collab_df['Relevancia (%)'] = collab_df['Relevancia (%)'].round(2)
                st.dataframe(
                    collab_df,
                    use_container_width=True,
                    column_config={
                        "Título": st.column_config.TextColumn("Título"),
                        "Relevancia (%)": st.column_config.NumberColumn("Relevancia (%)", format="%.2f")
                    }
                )
            except Exception as e:
                st.error(f"Error al generar recomendaciones: {str(e)}")
else:
    if submit_button:
        st.warning("Por favor, selecciona una película.")