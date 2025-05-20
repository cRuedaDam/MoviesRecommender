import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import time
import sys
import os

# Agregar el directorio raíz para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.recommenders.utils import get_poster_url

def parse_genres(genres_str):
    """
    Procesa una cadena de géneros y devuelve una cadena limpia y una lista de géneros.
    
    Args:
        genres_str (str): Cadena de géneros (e.g., '[{"name": "Action"}, {"name": "Comedy"}]').
        
    Returns:
        tuple: (cadena de géneros separados por espacio, lista de géneros).
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
    Preprocesa los datos de películas y enlaces.
    
    Args:
        metadata (pd.DataFrame): Datos de películas.
        links (pd.DataFrame): Datos de enlaces (movieId, tmdbId, etc.).
        
    Returns:
        tuple: (metadata procesado, matriz TF-IDF).
    """
    start_time = time.time()
    
    # Limpieza inicial
    metadata = metadata[metadata['id'].apply(lambda x: str(x).isdigit())].copy()
    metadata['id'] = metadata['id'].astype(int)
    metadata['title'] = metadata['title'].astype(str).str.strip()
    metadata['vote_average'] = pd.to_numeric(metadata['vote_average'], errors='coerce')
    metadata['vote_count'] = pd.to_numeric(metadata['vote_count'], errors='coerce')
    metadata['runtime'] = pd.to_numeric(metadata['runtime'], errors='coerce')
    metadata['release_date'] = pd.to_datetime(metadata['release_date'], errors='coerce')
    metadata['release_year'] = metadata['release_date'].dt.year
    metadata['imdb_id'] = metadata['imdb_id'].astype(str)

    links = links[links['movieId'].notna()].copy()
    links['movieId'] = links['movieId'].astype(int)
    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce').fillna(0).astype(int)

    # Intersección de IDs
    metadata_ids = set(metadata['id'])
    links_ids = set(links['tmdbId'])
    common_ids = metadata_ids.intersection(links_ids)
    print(f"Número de IDs en común: {len(common_ids)}")

    metadata = metadata[metadata['id'].isin(common_ids)]
    metadata = metadata.merge(links, left_on='id', right_on='tmdbId', how='inner')

    # Procesamiento de géneros
    metadata[['genres_clean', 'genres_list']] = pd.DataFrame(
        metadata['genres'].apply(parse_genres).tolist(), index=metadata.index
    )

    # Procesamiento de descripciones
    metadata['overview'] = metadata['overview'].fillna('')
    metadata['description'] = metadata['overview'] + ' ' + metadata['genres_clean']

    # Normalización de votos
    metadata['vote_count_norm'] = metadata['vote_count'].apply(
        lambda x: x / metadata['vote_count'].max() if pd.notna(x) else 0
    )
    metadata['vote_score'] = metadata.apply(
        lambda x: (x['vote_average'] * x['vote_count_norm']) if pd.notna(x['vote_average']) else 0, 
        axis=1
    )

    # Vectorización TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    tfidf_matrix = tfidf.fit_transform(metadata['overview'].fillna(''))

    # Selección de columnas
    metadata = metadata[[
        'id', 'movieId', 'imdb_id', 'title', 'genres', 'genres_clean', 'genres_list', 'overview',
        'description', 'belongs_to_collection', 'poster_path', 'vote_average', 'vote_count',
        'vote_count_norm', 'vote_score', 'runtime', 'release_date', 'release_year', 'original_language'
    ]]
    
    end_time = time.time()
    print(f"Tiempo de carga y preprocesamiento: {end_time - start_time:.2f} segundos")

    # Limpieza y verificación de posters
    metadata['poster_url'] = metadata['poster_path'].apply(
        lambda x: get_poster_url(x, verify=False) 
    )
    
    # Marcar posters válidos
    metadata['has_valid_poster'] = metadata['poster_url'].apply(
        lambda x: not x.endswith('Poster+no+disponible')
    )
    
    print(f"Posters válidos: {metadata['has_valid_poster'].sum()}/{len(metadata)}")

    print(metadata['poster_path'].isna().sum())
    print(metadata['poster_path'].head(10))
    print(metadata[['title', 'poster_path', 'poster_url']].sample(5))
    
    return metadata, tfidf_matrix