import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import time
import matplotlib.pyplot as plt
import seaborn as sns
from src.recommenders.utils import get_poster_url

def parse_genres(genres_str):
    """
    Convierte la cadena JSON con géneros en una cadena y lista de géneros.

    Parámetros:
        genres_str (str): Cadena en formato JSON con géneros, por ejemplo:
                          '[{"name": "Action"}, {"name": "Comedy"}]'.

    Retorna:
        tuple: Una cadena con nombres de géneros separados por espacios y una lista con esos géneros.
               Si no se puede procesar, retorna cadena vacía y lista vacía.
    """
    if pd.isna(genres_str) or genres_str is None:
        return "", []
    try:
        # Convierte la cadena JSON a lista de diccionarios
        genres = ast.literal_eval(str(genres_str))
        if isinstance(genres, list):
            # Extrae solo el campo 'name' de cada género válido
            genre_names = [g['name'] for g in genres if isinstance(g, dict) and 'name' in g]
            return " ".join(genre_names), genre_names
        return "", []
    except (ValueError, SyntaxError, TypeError) as e:
        # Si hay error en la conversión, se informa y retorna valores vacíos
        print(f"Error al procesar géneros: {e}, valor: {genres_str}")
        return "", []

def preprocess_data(metadata, links):
    """
    Limpia, transforma y prepara los datos de películas y enlaces para el sistema de recomendación.

    Parámetros:
        metadata (pd.DataFrame): DataFrame con información detallada de las películas.
        links (pd.DataFrame): DataFrame con enlaces de IDs entre diferentes bases de datos.

    Retorna:
        tuple: DataFrame de metadata procesado y matriz TF-IDF generada a partir de las descripciones.
    """
    start_time = time.time()

    # Filtra registros cuyo campo 'id' sea numérico para evitar datos inconsistentes
    metadata = metadata[metadata['id'].apply(lambda x: str(x).isdigit())].copy()
    metadata['id'] = metadata['id'].astype(int)

    # Limpieza y normalización de columnas principales
    metadata['title'] = metadata['title'].astype(str).str.strip()
    metadata['vote_average'] = pd.to_numeric(metadata['vote_average'], errors='coerce')
    metadata['vote_count'] = pd.to_numeric(metadata['vote_count'], errors='coerce')
    metadata['runtime'] = pd.to_numeric(metadata['runtime'], errors='coerce')
    metadata['release_date'] = pd.to_datetime(metadata['release_date'], errors='coerce')
    metadata['release_year'] = metadata['release_date'].dt.year
    metadata['imdb_id'] = metadata['imdb_id'].astype(str)

    # Limpieza y conversión de tipos en el DataFrame de enlaces
    links = links[links['movieId'].notna()].copy()
    links['movieId'] = links['movieId'].astype(int)
    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce').fillna(0).astype(int)

    # Identifica los IDs comunes entre ambos DataFrames para enlazar correctamente la información
    metadata_ids = set(metadata['id'])
    links_ids = set(links['tmdbId'])
    common_ids = metadata_ids.intersection(links_ids)
    print(f"Número de IDs en común: {len(common_ids)}")

    # Filtra metadata para que contenga solo las películas con enlaces válidos
    metadata = metadata[metadata['id'].isin(common_ids)]

    # Fusiona metadata y links para agregar información extra de enlaces
    metadata = metadata.merge(links, left_on='id', right_on='tmdbId', how='inner')

    # Aplica la función para extraer y limpiar géneros
    metadata[['genres_clean', 'genres_list']] = pd.DataFrame(
        metadata['genres'].apply(parse_genres).tolist(), index=metadata.index
    )

    # Combina la descripción y géneros para formar la descripción completa
    metadata['overview'] = metadata['overview'].fillna('')
    metadata['description'] = metadata['overview'] + ' ' + metadata['genres_clean']

    # Normaliza la cantidad de votos para ponderar la relevancia
    metadata['vote_count_norm'] = metadata['vote_count'].apply(
        lambda x: x / metadata['vote_count'].max() if pd.notna(x) else 0
    )

    # Calcula un score de votación ponderado por cantidad y promedio
    metadata['vote_score'] = metadata.apply(
        lambda x: (x['vote_average'] * x['vote_count_norm']) if pd.notna(x['vote_average']) else 0, 
        axis=1
    )

    # Genera matriz TF-IDF para la descripción, limitando a 3000 características y eliminando stopwords en inglés
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    tfidf_matrix = tfidf.fit_transform(metadata['overview'].fillna(''))

    # Selecciona columnas relevantes para análisis y visualización
    metadata = metadata[[
        'id', 'movieId', 'imdb_id', 'title', 'genres', 'genres_clean', 'genres_list', 'overview',
        'description', 'belongs_to_collection', 'poster_path', 'vote_average', 'vote_count',
        'vote_count_norm', 'vote_score', 'runtime', 'release_date', 'release_year', 'original_language'
    ]]

    end_time = time.time()
    print(f"Tiempo de carga y preprocesamiento: {end_time - start_time:.2f} segundos")

    # Construye las URLs para los posters usando función externa sin validar si existen realmente
    metadata['poster_url'] = metadata['poster_path'].apply(
        lambda x: get_poster_url(x, verify=False) 
    )

    # Identifica qué posters son válidos comparando si la URL no apunta a un placeholder
    metadata['has_valid_poster'] = metadata['poster_url'].apply(
        lambda x: not x.endswith('Poster+no+disponible')
    )

    # Estadísticas básicas para validar datos de posters
    print(f"Posters válidos: {metadata['has_valid_poster'].sum()}/{len(metadata)}")
    print(metadata['poster_path'].isna().sum())
    print(metadata['poster_path'].head(10))
    print(metadata[['title', 'poster_path', 'poster_url']].sample(5))

    return metadata, tfidf_matrix
