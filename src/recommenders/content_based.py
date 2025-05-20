from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import pandas as pd
import ast
import time
from src.recommenders.utils import find_movie_title

def content_based_recommender(input_title, metadata, n_recommendations=10, tfidf_matrix=None):
    """
    Genera recomendaciones basadas en contenido para una película dada, con mayor énfasis en géneros y sagas
    
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
    print(f"Película de entrada: {input_title}, Año: {movie_year}, Géneros: {', '.join(movie_genres)}")
    
    # Calcular similitud coseno
    cosine_similarities = linear_kernel(tfidf_matrix[movie_idx:movie_idx+1], tfidf_matrix).flatten()
    
    # Crear un DataFrame para manejar las similitudes
    similarity_df = pd.DataFrame({
        'index': range(len(cosine_similarities)),
        'title': metadata['title'],
        'similarity': cosine_similarities * 100,  # Convertir a porcentaje
        'genres_list': metadata['genres_list'],
        'vote_count': metadata['vote_count'],
        'vote_average': metadata['vote_average'],
        'release_year': metadata['release_year'],
        'belongs_to_collection': metadata['belongs_to_collection'],
        'original_language': metadata['original_language']
    })
    
    # Filtrar la película de entrada
    filtered_df = similarity_df[similarity_df['title'] != input_title].copy()
    
    # 1. Factor de colección (prioridad máxima si pertenecen a la misma saga)
    filtered_df['collection_match'] = filtered_df['belongs_to_collection'].apply(
        lambda x: 1 if pd.notna(x) and pd.notna(movie_collection) and x == movie_collection else 0
    )
    
    # 2. Factor de similitud de género (0-1) - MEJORADO
    filtered_df['genre_match'] = filtered_df['genres_list'].apply(
        lambda x: len(set(x).intersection(movie_genres)) / max(len(movie_genres), 1)
    )
    
    # 3. Factor de coincidencia exacta de género principal (0-1)
    main_genre = next(iter(movie_genres), None) if movie_genres else None
    filtered_df['main_genre_match'] = filtered_df['genres_list'].apply(
        lambda x: 1 if main_genre and main_genre in x else 0
    )
    
    # 4. Factor de cercanía temporal (0-1)
    MAX_YEAR_DIFF = 10  # Más estricto para mayor relevancia
    filtered_df['year_proximity'] = filtered_df['release_year'].apply(
        lambda x: max(0, 1 - abs(x - movie_year) / MAX_YEAR_DIFF) if pd.notna(x) and pd.notna(movie_year) else 0.2
    )

    # 5. Factor de idioma (0-1)
    filtered_df['language_match'] = filtered_df['original_language'].apply(
        lambda x: 1 if x == movie_language else 0.2
    )
    
    # 6. Factor de popularidad (0-1)
    max_votes = filtered_df['vote_count'].max()
    filtered_df['popularity_factor'] = filtered_df['vote_count'].apply(
        lambda x: 0.2 + 0.8 * (np.log1p(x) / np.log1p(max_votes)) if x > 0 else 0.2
    )
    
    # 7. Factor de calificación (0-1)
    filtered_df['rating_factor'] = filtered_df['vote_average'].apply(
        lambda x: 0.2 + 0.8 * (x / 10) if x > 0 else 0.2
    )
    
    # Ajustar pesos (priorizando colección y género)
    SIMILARITY_WEIGHT = 0.3    # Reducido para dar más peso a otros factores
    COLLECTION_WEIGHT = 0.15   # Peso alto para películas de la misma saga
    GENRE_WEIGHT = 0.1         # Aumentado para mejor matching de géneros
    MAIN_GENRE_WEIGHT = 0.15   # Peso adicional para género principal
    YEAR_WEIGHT = 0.05
    LANGUAGE_WEIGHT = 0.05
    POPULARITY_WEIGHT = 0.1
    RATING_WEIGHT = 0.05
    
    # Calcular score combinado
    filtered_df['combined_score'] = (
        SIMILARITY_WEIGHT * filtered_df['similarity'] + 
        COLLECTION_WEIGHT * (filtered_df['collection_match'] * 100) +
        GENRE_WEIGHT * (filtered_df['genre_match'] * 100) +
        MAIN_GENRE_WEIGHT * (filtered_df['main_genre_match'] * 100) +
        YEAR_WEIGHT * (filtered_df['year_proximity'] * 100) +
        LANGUAGE_WEIGHT * (filtered_df['language_match'] * 100) +
        POPULARITY_WEIGHT * (filtered_df['popularity_factor'] * 100) +
        RATING_WEIGHT * (filtered_df['rating_factor'] * 100)
    )
    
    # Ordenar primero por colección, luego por coincidencia de género principal, luego por puntuación
    sorted_df = filtered_df.sort_values(
        ['collection_match', 'main_genre_match', 'combined_score'], 
        ascending=[False, False, False]
    )
    
    # Seleccionar las top n recomendaciones
    top_recommendations = sorted_df.head(n_recommendations)
    
    # Convertir a formato de lista de tuplas (título, similitud)
    recommendations = list(zip(top_recommendations['title'], top_recommendations['combined_score']))
    
    print(f"Tiempo de ejecución de Content Based: {time.time() - start_time:.2f} segundos")
    return recommendations