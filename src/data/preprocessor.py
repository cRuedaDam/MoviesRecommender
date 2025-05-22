import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import time
import os
from src.recommenders.utils import get_poster_url

def parse_genres(genres_str):
    """
    Parsea una cadena en formato JSON que representa géneros y devuelve dos representaciones:
    una cadena unificada con los nombres de los géneros y una lista de dichos nombres.

    Parameters:
        genres_str (str): Cadena con representación de lista de diccionarios de géneros.

    Returns:
        tuple[str, list[str]]: Géneros como cadena unificada y lista individual.
    """
    if pd.isna(genres_str) or genres_str is None:
        return "", []
    try:
        genres = ast.literal_eval(str(genres_str))
        if isinstance(genres, list):
            genre_names = [g['name'] for g in genres if isinstance(g, dict) and 'name' in g]
            return " ".join(genre_names), genre_names
        return "", []
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Error al procesar géneros: {e}, valor: {genres_str}")
        return "", []

def preprocess_data(metadata, links):
    """
    Realiza el preprocesamiento completo de los datos de películas y enlaces. Este paso incluye:
    limpieza, normalización de columnas, fusión de DataFrames, generación de descripciones enriquecidas
    y cálculo de matriz TF-IDF.

    Parameters:
        metadata (pd.DataFrame): Información detallada de películas, incluyendo descripción, géneros, etc.
        links (pd.DataFrame): Correspondencia entre IDs de distintas fuentes (MovieLens, TMDB, IMDB).

    Returns:
        tuple[pd.DataFrame, scipy.sparse.csr_matrix]: 
            DataFrame procesado y matriz TF-IDF correspondiente a las descripciones enriquecidas.
    """
    start_time = time.time()

    # Filtrado y tipado de IDs
    metadata = metadata[metadata['id'].apply(lambda x: str(x).isdigit())].copy()
    metadata['id'] = metadata['id'].astype(int)

    # Normalización de campos clave
    metadata['title'] = metadata['title'].astype(str).str.strip()
    metadata['vote_average'] = pd.to_numeric(metadata['vote_average'], errors='coerce')
    metadata['vote_count'] = pd.to_numeric(metadata['vote_count'], errors='coerce')
    metadata['runtime'] = pd.to_numeric(metadata['runtime'], errors='coerce')
    metadata['release_date'] = pd.to_datetime(metadata['release_date'], errors='coerce')
    metadata['release_year'] = metadata['release_date'].dt.year
    metadata['imdb_id'] = metadata['imdb_id'].astype(str)

    # Limpieza del DataFrame de enlaces
    links = links[links['movieId'].notna()].copy()
    links['movieId'] = links['movieId'].astype(int)
    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce').fillna(0).astype(int)

    # Intersección de IDs comunes
    metadata_ids = set(metadata['id'])
    links_ids = set(links['tmdbId'])
    common_ids = metadata_ids.intersection(links_ids)
    print(f"Número de IDs en común: {len(common_ids)}")

    # Filtrado y fusión
    metadata = metadata[metadata['id'].isin(common_ids)]
    metadata = metadata.merge(links, left_on='id', right_on='tmdbId', how='inner')

    # Procesamiento de géneros
    metadata[['genres_clean', 'genres_list']] = pd.DataFrame(
        metadata['genres'].apply(parse_genres).tolist(), index=metadata.index
    )

    # Enriquecimiento de la descripción
    metadata['overview'] = metadata['overview'].fillna('')
    metadata['description'] = metadata['overview'] + ' ' + metadata['genres_clean']

    # Normalización de votos
    metadata['vote_count_norm'] = metadata['vote_count'].apply(
        lambda x: x / metadata['vote_count'].max() if pd.notna(x) else 0
    )

    # Cálculo de puntuación ponderada
    metadata['vote_score'] = metadata.apply(
        lambda x: (x['vote_average'] * x['vote_count_norm']) if pd.notna(x['vote_average']) else 0, 
        axis=1
    )

    # Matriz TF-IDF basada en descripciones enriquecidas
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    tfidf_matrix = tfidf.fit_transform(metadata['overview'].fillna(''))

    # Selección de columnas relevantes para el sistema
    metadata = metadata[[
        'id', 'movieId', 'imdb_id', 'title', 'genres', 'genres_clean', 'genres_list', 'overview',
        'description', 'belongs_to_collection', 'poster_path', 'vote_average', 'vote_count',
        'vote_count_norm', 'vote_score', 'runtime', 'release_date', 'release_year', 'original_language',
        'tmdbId'
    ]]

    end_time = time.time()
    print(f"Tiempo de carga y preprocesamiento: {end_time - start_time:.2f} segundos")

    return metadata, tfidf_matrix
