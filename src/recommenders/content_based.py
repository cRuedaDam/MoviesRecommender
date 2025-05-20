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
    
    # Comprobar si hay datos de producción y director/actores disponibles
    has_production = 'production_companies' in metadata.columns
    has_crew = 'director' in metadata.columns
    has_cast = 'cast' in metadata.columns
    
    # Extraer información de producción si está disponible
    movie_production_companies = []
    if has_production and pd.notna(metadata.loc[movie_idx, 'production_companies']):
        try:
            movie_production_companies = [
                company['name'] for company in ast.literal_eval(metadata.loc[movie_idx, 'production_companies'])
            ]
        except (ValueError, KeyError):
            movie_production_companies = []
    
    # Extraer información de director si está disponible
    movie_director = None
    if has_crew and pd.notna(metadata.loc[movie_idx, 'director']):
        movie_director = metadata.loc[movie_idx, 'director']
    
    # Extraer información de actores principales si está disponible
    movie_cast = []
    if has_cast and pd.notna(metadata.loc[movie_idx, 'cast']):
        try:
            movie_cast = ast.literal_eval(metadata.loc[movie_idx, 'cast'])[:3]  # Primeros 3 actores
        except (ValueError, KeyError):
            movie_cast = []
    
    # Imprimir información de la película
    print(f"Película de entrada: {input_title}, Año: {movie_year}, Géneros: {movie_genres}")
    if movie_production_companies:
        print(f"Productoras: {movie_production_companies}")
    if movie_director:
        print(f"Director: {movie_director}")
    if movie_cast:
        print(f"Actores principales: {', '.join(movie_cast)}")
    
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
    
    # Añadir columnas condicionales si están disponibles
    if has_production:
        similarity_df['production_companies'] = metadata['production_companies']
    if has_crew:
        similarity_df['director'] = metadata['director']
    if has_cast:
        similarity_df['cast'] = metadata['cast']
    
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
    
    # 3. Factor de productora común (0-1)
    filtered_df['production_match'] = 0.0
    if has_production and movie_production_companies:
        filtered_df['production_match'] = filtered_df['production_companies'].apply(
            lambda x: calculate_production_match(x, movie_production_companies) if pd.notna(x) else 0.0
        )
    
    # 4. Factor de director común (0-1)
    filtered_df['director_match'] = 0.0
    if has_crew and movie_director:
        filtered_df['director_match'] = filtered_df['director'].apply(
            lambda x: 1.0 if pd.notna(x) and x == movie_director else 0.0
        )
    
    # 5. Factor de actores comunes (0-1)
    filtered_df['cast_match'] = 0.0
    if has_cast and movie_cast:
        filtered_df['cast_match'] = filtered_df['cast'].apply(
            lambda x: calculate_cast_match(x, movie_cast) if pd.notna(x) else 0.0
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
    PRODUCTION_WEIGHT = 0.05
    DIRECTOR_WEIGHT = 0.05
    CAST_WEIGHT = 0.05
    
    # Ajustar los pesos según género de la película
    # Para animación, dar más peso a la productora y género
    if 'Animation' in movie_genres:
        GENRE_WEIGHT = 0.25
        PRODUCTION_WEIGHT = 0.15
        SIMILARITY_WEIGHT = 0.25
        YEAR_WEIGHT = 0.15
        POPULARITY_WEIGHT = 0.1
        RATING_WEIGHT = 0.1
        DIRECTOR_WEIGHT = 0.0
        CAST_WEIGHT = 0.0
    
    # Para películas de franquicias (que pertenecen a colecciones), dar más peso a la productora
    if pd.notna(movie_collection):
        PRODUCTION_WEIGHT = 0.15
        SIMILARITY_WEIGHT = 0.25
    
    # Calcular score combinado
    filtered_df['combined_score'] = (
        SIMILARITY_WEIGHT * filtered_df['similarity'] + 
        GENRE_WEIGHT * (filtered_df['genre_match'] * 100) +
        YEAR_WEIGHT * (filtered_df['year_proximity'] * 100) +
        POPULARITY_WEIGHT * (filtered_df['popularity_factor'] * 100) +
        RATING_WEIGHT * (filtered_df['rating_factor'] * 100) +
        PRODUCTION_WEIGHT * (filtered_df['production_match'] * 100) +
        DIRECTOR_WEIGHT * (filtered_df['director_match'] * 100) +
        CAST_WEIGHT * (filtered_df['cast_match'] * 100)
    )
    
    # Ordenar por puntuación combinada (de mayor a menor)
    sorted_df = filtered_df.sort_values('combined_score', ascending=False)
    
    # Seleccionar las top n recomendaciones
    top_recommendations = sorted_df.head(n_recommendations)
    
    # Convertir a formato de lista de tuplas (título, similitud)
    recommendations = list(zip(top_recommendations['title'], top_recommendations['combined_score']))
    
    print(f"Tiempo de ejecución de Content Based: {time.time() - start_time:.2f} segundos")
    return recommendations


def calculate_production_match(production_str, target_companies):
    """Calcula la similitud entre compañías de producción"""
    try:
        if isinstance(production_str, str):
            companies = ast.literal_eval(production_str)
            if isinstance(companies, list):
                movie_companies = [company['name'] for company in companies if 'name' in company]
                common_companies = set(movie_companies).intersection(set(target_companies))
                return len(common_companies) / max(len(target_companies), len(movie_companies)) if movie_companies else 0
    except (ValueError, TypeError, KeyError):
        pass
    return 0.0


def calculate_cast_match(cast_str, target_cast):
    """Calcula la similitud entre elencos"""
    try:
        if isinstance(cast_str, str):
            cast = ast.literal_eval(cast_str)
            if isinstance(cast, list):
                cast = cast[:3]  # Considerar solo los 3 primeros actores
                common_cast = set(cast).intersection(set(target_cast))
                return len(common_cast) / max(len(target_cast), len(cast)) if cast else 0
    except (ValueError, TypeError):
        pass
    return 0.0