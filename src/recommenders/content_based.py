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
    
    # Validar que el título de entrada sea una cadena no vacía
    if not isinstance(input_title, str) or not input_title.strip():
        raise ValueError("El título de entrada debe ser una cadena no vacía.")
    
    # Comprobar si el título está en el dataset; si no, buscar título similar
    if input_title not in metadata['title'].values:
        closest_title = find_movie_title(input_title, metadata['title'].values)
        if closest_title:
            print(f"No se encontró '{input_title}'. ¿Quizás se refiere a '{closest_title}'?")
            input_title = closest_title
        else:
            raise ValueError(f"No se puede generar recomendaciones para '{input_title}' porque no se encontró en el dataset.")
    
    # Obtener índice y metadatos relevantes de la película de entrada
    movie_idx = metadata[metadata['title'] == input_title].index[0]
    movie_id = metadata.loc[movie_idx, 'movieId']
    movie_genres = set(metadata.loc[movie_idx, 'genres_list'])
    movie_year = metadata.loc[movie_idx, 'release_year']
    movie_language = metadata.loc[movie_idx, 'original_language']
    movie_collection = metadata.loc[movie_idx, 'belongs_to_collection']
    
    # Mostrar información básica de la película de entrada
    print(f"Película de entrada: {input_title}, Año: {movie_year}, Géneros: {', '.join(movie_genres)}")
    
    # Calcular similitud coseno entre la película de entrada y todas las demás usando la matriz TF-IDF
    cosine_similarities = linear_kernel(tfidf_matrix[movie_idx:movie_idx+1], tfidf_matrix).flatten()
    
    # Crear DataFrame para manejar todas las películas con su similitud y metadatos necesarios
    similarity_df = pd.DataFrame({
        'index': range(len(cosine_similarities)),
        'title': metadata['title'],
        'similarity': cosine_similarities * 100,  # Convertir similitud a porcentaje para mejor interpretación
        'genres_list': metadata['genres_list'],
        'vote_count': metadata['vote_count'],
        'vote_average': metadata['vote_average'],
        'release_year': metadata['release_year'],
        'belongs_to_collection': metadata['belongs_to_collection'],
        'original_language': metadata['original_language']
    })
    
    # Excluir la película de entrada del listado de recomendaciones
    filtered_df = similarity_df[similarity_df['title'] != input_title].copy()
    
    # 1. Factor colección: Priorizar películas que pertenezcan a la misma saga/colección
    filtered_df['collection_match'] = filtered_df['belongs_to_collection'].apply(
        lambda x: 1 if pd.notna(x) and pd.notna(movie_collection) and x == movie_collection else 0
    )
    
    # 2. Factor similitud de género: Proporción de géneros en común respecto a la película de entrada
    filtered_df['genre_match'] = filtered_df['genres_list'].apply(
        lambda x: len(set(x).intersection(movie_genres)) / max(len(movie_genres), 1)
    )
    
    # 3. Factor coincidencia género principal: Si el género principal coincide, asignar 1, sino 0
    main_genre = next(iter(movie_genres), None) if movie_genres else None
    filtered_df['main_genre_match'] = filtered_df['genres_list'].apply(
        lambda x: 1 if main_genre and main_genre in x else 0
    )
    
    # 4. Factor proximidad temporal: Más cercanía en años da un factor más alto (0 a 1)
    MAX_YEAR_DIFF = 20  # Diferencia máxima de años para considerar proximidad
    filtered_df['year_proximity'] = filtered_df['release_year'].apply(
        lambda x: max(0, 1 - abs(x - movie_year) / MAX_YEAR_DIFF) if pd.notna(x) and pd.notna(movie_year) else 0.2
    )
    
    # 5. Factor idioma: Preferencia a películas en el mismo idioma original
    filtered_df['language_match'] = filtered_df['original_language'].apply(
        lambda x: 1 if x == movie_language else 0.2
    )
    
    # 6. Factor popularidad: Popularidad basada en votos, normalizada y suavizada para evitar 0
    max_votes = filtered_df['vote_count'].max()
    filtered_df['popularity_factor'] = filtered_df['vote_count'].apply(
        lambda x: 0.2 + 0.8 * (np.log1p(x) / np.log1p(max_votes)) if x > 0 else 0.2
    )
    
    # 7. Factor calificación: Calificación promedio, normalizada entre 0.2 y 1
    filtered_df['rating_factor'] = filtered_df['vote_average'].apply(
        lambda x: 0.2 + 0.8 * (x / 10) if x > 0 else 0.2
    )
    
    # Pesos para combinar los factores en un solo score, priorizando colección y género
    SIMILARITY_WEIGHT = 0.3
    COLLECTION_WEIGHT = 0.15
    GENRE_WEIGHT = 0.1
    MAIN_GENRE_WEIGHT = 0.15
    YEAR_WEIGHT = 0.05
    LANGUAGE_WEIGHT = 0.05
    POPULARITY_WEIGHT = 0.1
    RATING_WEIGHT = 0.05

    print("Maximo valor de 'similarity':")
    print(filtered_df['similarity'].max())
    
    # Calcular score combinado para ranking final
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
    
    # Ordenar por colección (desc), género principal (desc) y score combinado (desc) para mayor relevancia
    sorted_df = filtered_df.sort_values(
        ['collection_match', 'main_genre_match', 'combined_score'], 
        ascending=[False, False, False]
    )
    
    # Seleccionar las top n películas recomendadas según el ranking
    top_recommendations = sorted_df.head(n_recommendations)
    
    # Convertir las recomendaciones a lista de tuplas (título, score combinado)
    recommendations = list(zip(top_recommendations['title'], top_recommendations['combined_score']))
    
    print(f"Tiempo de ejecución de Content Based: {time.time() - start_time:.2f} segundos")
    return recommendations


def evaluate_content_based(metadata, tfidf_matrix, content_based_recommender, n_recommendations=10, test_samples=5, k=10):
    """
    Evalúa el recomendador content_based usando un conjunto de prueba simple basado en género compartido.

    Args:
        metadata (pd.DataFrame): DataFrame con metadatos de películas.
        tfidf_matrix (scipy.sparse.csr_matrix): Matriz TF-IDF usada para la similitud.
        content_based_recommender (function): Función de recomendación que acepta (input_title, metadata, n_recommendations, tfidf_matrix).
        n_recommendations (int): Número de recomendaciones a generar por película.
        test_samples (int): Número de películas del conjunto de prueba a evaluar.
        k (int): Valor para Precision@K y Recall@K.

    Returns:
        dict: Diccionario con métricas promedio Precision@K y Recall@K.
    """

    def precision_at_k(recommended, relevant, k):
        recommended_at_k = recommended[:k]
        return len(set(recommended_at_k).intersection(relevant)) / k

    def recall_at_k(recommended, relevant, k):
        recommended_at_k = recommended[:k]
        if len(relevant) == 0:
            return 0
        return len(set(recommended_at_k).intersection(relevant)) / len(relevant)

    precisions = []
    recalls = []

    # Seleccionar aleatoriamente películas de prueba
    test_movies = metadata.sample(n=test_samples, random_state=42)

    for _, row in test_movies.iterrows():
        input_title = row['title']
        input_genres = set(row['genres_list'])

        # Definir películas relevantes: aquellas que comparten género con la de entrada, excluyendo la misma
        relevant_movies = set(
            metadata[
                metadata['genres_list'].apply(lambda genres: len(input_genres.intersection(genres)) > 0) &
                (metadata['title'] != input_title)
            ]['title']
        )

        # Obtener recomendaciones
        recs = content_based_recommender(input_title, metadata, n_recommendations=n_recommendations, tfidf_matrix=tfidf_matrix)
        recommended_titles = [title for title, score in recs]

        # Calcular métricas
        prec = precision_at_k(recommended_titles, relevant_movies, k)
        rec = recall_at_k(recommended_titles, relevant_movies, k)

        precisions.append(prec)
        recalls.append(rec)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0

    return {
        'Average Precision@K': avg_precision,
        'Average Recall@K': avg_recall
    }
