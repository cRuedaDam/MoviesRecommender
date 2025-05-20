from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import time
from src.recommenders.utils import find_movie_title, add_diversity

def collaborative_recommender(user_id, ratings, metadata, n_recommendations=10, input_title=None, tfidf_matrix=None):
    """
    Genera recomendaciones colaborativas basadas en filtrado colaborativo.
    
    Args:
        user_id (int): ID del usuario.
        ratings (pd.DataFrame): DataFrame con calificaciones.
        metadata (pd.DataFrame): DataFrame con metadatos de películas.
        n_recommendations (int): Número de recomendaciones a devolver.
        input_title (str, optional): Título de la película de entrada para hibridación.
        tfidf_matrix (scipy.sparse.csr_matrix, optional): Matriz TF-IDF para hibridación.
    
    Returns:
        list: Lista de tuplas (título, relevancia) con las recomendaciones.
    """
    start_time = time.time()
    
    # Validar entrada
    if user_id not in ratings['userId'].unique():
        raise ValueError(f"El usuario {user_id} no se encuentra en los datos de calificaciones.")
    
    # Crear matriz usuario-película
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    user_item_matrix_sparse = csr_matrix(user_item_matrix.values)
    
    # Calcular similitud entre usuarios
    user_similarity = cosine_similarity(user_item_matrix_sparse)
    user_idx = user_item_matrix.index.get_loc(user_id)
    similar_users = user_similarity[user_idx].argsort()[::-1][1:]  # Excluir al propio usuario
    
    # Obtener películas no vistas por el usuario
    user_ratings = user_item_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index
    
    # Predecir calificaciones
    predicted_ratings = []
    for movie_id in unrated_movies:
        similar_ratings = ratings[(ratings['movieId'] == movie_id) & 
                                (ratings['userId'].isin(user_item_matrix.index[similar_users]))]
        if not similar_ratings.empty:
            weighted_sum = sum(user_similarity[user_idx, user_item_matrix.index.get_loc(u)] * r 
                             for u, r in similar_ratings[['userId', 'rating']].values)
            similarity_sum = sum(abs(user_similarity[user_idx, user_item_matrix.index.get_loc(u)]) 
                               for u in similar_ratings['userId'])
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                predicted_ratings.append((movie_id, predicted_rating))
    
    # Ordenar por calificación predicha
    predicted_ratings.sort(key=lambda x: x[1], reverse=True)
    top_movies = predicted_ratings[:n_recommendations]
    
    # Convertir movieId a títulos
    recommendations = []
    for movie_id, score in top_movies:
        title = find_movie_title(movie_id, metadata)
        if title:
            recommendations.append((title, score * 20))  # Escalar a 0-100
    
    # Hibridación con contenido si input_title está proporcionado
    if input_title:
        if not isinstance(input_title, str) or not input_title.strip():
            raise ValueError("El título de entrada debe ser una cadena no vacía.")
        if input_title not in metadata['title'].values:
            raise ValueError(f"No se puede generar recomendaciones para '{input_title}' porque no se encontró en el dataset.")
        if tfidf_matrix is not None:
            movie_idx = metadata[metadata['title'] == input_title].index[0]
            cosine_similarities = linear_kernel(tfidf_matrix[movie_idx:movie_idx+1], tfidf_matrix).flatten()
            similar_indices = cosine_similarities.argsort()[-(n_recommendations + 1):-1][::-1]
            
            content_recs = [(metadata.iloc[idx]['title'], cosine_similarities[idx] * 100) 
                           for idx in similar_indices if metadata.iloc[idx]['title'] != input_title]
            
            # Combinar recomendaciones
            combined = {}
            for title, score in recommendations:
                combined[title] = combined.get(title, 0) + score * 0.7
            for title, score in content_recs:
                combined[title] = combined.get(title, 0) + score * 0.3
            
            recommendations = [(title, score) for title, score in combined.items()]
            recommendations.sort(key=lambda x: x[1], reverse=True)
            recommendations = recommendations[:n_recommendations]
    
    # Aplicar diversidad
    recommendations = add_diversity(recommendations, metadata, n_recommendations)
    
    print(f"Tiempo de ejecución de Collaborative: {time.time() - start_time:.2f} segundos")
    return recommendations