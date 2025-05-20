from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import pandas as pd
import ast
import time
from src.recommenders.utils import find_movie_title

def content_based_recommender(input_title, metadata, n_recommendations=10, tfidf_matrix=None):
    """
    Genera recomendaciones basadas en contenido para una película dada
    
    Args:
        input_title (str): Título de la película de entrada.
        metadata (pd.DataFrame): DataFrame con metadatos de películas.
        n_recommendations (int): Número de recomendaciones a devolver.
        tfidf_matrix (scipy.sparse.csr_matrix): Matriz TF-IDF de características.
    
    Returns:
        list: Lista de tuplas (título, similitud) con las recomendaciones.
    """
    start_time = time.time()
    
    # Validar entrada
    if not isinstance(input_title, str) or not input_title.strip():
        raise ValueError("El título de entrada debe ser una cadena no vacía.")
    
    if input_title not in metadata['title'].values:
        closest_title = find_movie_title(input_title, metadata['title'].values)
        if closest_title:
            print(f"No se encontró '{input_title}'. ¿Quizás se refiere a '{closest_title}'?")
            input_title = closest_title
        else:
            raise ValueError(f"No se puede generar recomendaciones para '{input_title}' porque no se encontró en el dataset.")
    
    # Obtener información de la película de entrada
    movie_idx = metadata[metadata['title'] == input_title].index[0]
    movie_id = metadata.loc[movie_idx, 'movieId']
    movie_genres = set(metadata.loc[movie_idx, 'genres_list'])
    movie_year = metadata.loc[movie_idx, 'release_year']
    movie_language = metadata.loc[movie_idx, 'original_language']
    movie_collection = metadata.loc[movie_idx, 'belongs_to_collection']
    
    # Imprimir información de la película
    print(f"Película de entrada: {input_title}, Año: {movie_year}, Géneros: {movie_genres}")
    
    # Calcular similitud coseno
    cosine_similarities = linear_kernel(tfidf_matrix[movie_idx:movie_idx+1], tfidf_matrix).flatten()
    
    # Crear un DataFrame para manejar las similitudes
    similarity_df = pd.DataFrame({
        'index': range(len(cosine_similarities)),
        'title': metadata['title'],
        'similarity': cosine_similarities * 100,  # Convertir a porcentaje
        'genres_list': metadata['genres_list'],
        'vote_count': metadata['vote_count'],
        'vote_average': metadata['vote_average'] if 'vote_average' in metadata.columns else 5.0,
        'release_year': metadata['release_year']
    })
    
    # Filtrar la película de entrada
    filtered_df = similarity_df[similarity_df['title'] != input_title]
    
    # Calcular factores adicionales de similitud
    
    # 1. Factor de similitud de género (0-1)
    filtered_df['genre_match'] = filtered_df['genres_list'].apply(
        lambda x: len(set(x).intersection(movie_genres)) / max(len(movie_genres), len(x))
    )
    
    # 2. Factor de cercanía temporal (0-1)
    MAX_YEAR_DIFF = 20  # Máxima diferencia de años para considerar
    filtered_df['year_proximity'] = filtered_df['release_year'].apply(
        lambda x: max(0, 1 - abs(x - movie_year) / MAX_YEAR_DIFF) if pd.notna(x) and pd.notna(movie_year) else 0.5
    )

    # Calcular factor de popularidad (0-1)
    max_votes = filtered_df['vote_count'].max()
    filtered_df['popularity_factor'] = filtered_df['vote_count'].apply(
        lambda x: 0.5 + 0.5 * (np.log1p(x) / np.log1p(max_votes)) if x > 0 else 0.5
    )
    
    # Calcular factor de calificación (0-1)
    filtered_df['rating_factor'] = filtered_df['vote_average'].apply(
        lambda x: 0.5 + 0.5 * (x / 10) if x > 0 else 0.5
    )
    
    # Ajustar pesos según el tipo de película
    SIMILARITY_WEIGHT = 0.3
    GENRE_WEIGHT = 0.2
    YEAR_WEIGHT = 0.1
    POPULARITY_WEIGHT = 0.15
    RATING_WEIGHT = 0.1
    
    # Calcular score combinado
    filtered_df['combined_score'] = (
        SIMILARITY_WEIGHT * filtered_df['similarity'] + 
        GENRE_WEIGHT * (filtered_df['genre_match'] * 100) +
        YEAR_WEIGHT * (filtered_df['year_proximity'] * 100) +
        POPULARITY_WEIGHT * (filtered_df['popularity_factor'] * 100) +
        RATING_WEIGHT * (filtered_df['rating_factor'] * 100)
    )
    
    # Ordenar por puntuación combinada (de mayor a menor)
    sorted_df = filtered_df.sort_values('combined_score', ascending=False)
    
    # Seleccionar las top n recomendaciones
    top_recommendations = sorted_df.head(n_recommendations)
    
    # Convertir a formato de lista de tuplas (título, similitud)
    recommendations = list(zip(top_recommendations['title'], top_recommendations['combined_score']))
    
    print(f"Tiempo de ejecución de Content Based: {time.time() - start_time:.2f} segundos")
    return recommendations