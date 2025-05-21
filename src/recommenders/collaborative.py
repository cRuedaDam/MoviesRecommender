from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import time
import re
from src.recommenders.utils import find_movie_title, add_diversity

def find_movie_improved(input_title, metadata, verbose=True):
    """
    Busca una película en el dataset de metadata con métodos mejorados.
    """
    if not input_title or input_title.strip() == "":
        if verbose:
            print("Título de entrada vacío")
        return None
    
    # Normalizar el título de entrada
    input_title_norm = input_title.lower().strip()
    
    # 1. Búsqueda exacta
    exact_match = metadata[metadata['title'].str.lower() == input_title_norm]
    if not exact_match.empty:
        movie_id = exact_match.iloc[0]['movieId']
        if verbose:
            print(f"Coincidencia exacta encontrada para '{input_title}': {exact_match.iloc[0]['title']} (ID: {movie_id})")
        return movie_id
    
    # 2. Búsqueda sin año (eliminando patrón tipo (1972))
    def remove_year(title):
        return re.sub(r"\s*\(\d{4}\)\s*$", "", title).lower()
    
    titles_no_year = metadata['title'].apply(remove_year)
    year_matches = metadata[titles_no_year == input_title_norm]
    if not year_matches.empty:
        movie_id = year_matches.iloc[0]['movieId']
        if verbose:
            print(f"Coincidencia sin año encontrada para '{input_title}': {year_matches.iloc[0]['title']} (ID: {movie_id})")
        return movie_id
    
    # 3. Búsqueda por contiene
    contains_matches = metadata[metadata['title'].str.lower().str.contains(input_title_norm, regex=False)]
    if not contains_matches.empty:
        sorted_matches = contains_matches.sort_values(by=['vote_average', 'vote_count'], ascending=False)
        movie_id = sorted_matches.iloc[0]['movieId']
        if verbose:
            print(f"Coincidencia parcial encontrada para '{input_title}': {sorted_matches.iloc[0]['title']} (ID: {movie_id})")
        return movie_id
    
    # 4. Búsqueda por palabras clave
    keywords = input_title_norm.split()
    if len(keywords) > 0:
        for keyword in keywords:
            if len(keyword) > 3:  # Ignoramos palabras cortas como "the", "of", etc.
                matches = metadata[metadata['title'].str.lower().str.contains(keyword)]
                if not matches.empty:
                    sorted_matches = matches.sort_values(by=['vote_average', 'vote_count'], ascending=False)
                    movie_id = sorted_matches.iloc[0]['movieId']
                    if verbose:
                        print(f"Coincidencia por palabra clave '{keyword}' encontrada para '{input_title}': {sorted_matches.iloc[0]['title']} (ID: {movie_id})")
                    return movie_id
    
    if verbose:
        print(f"No se encontró ninguna coincidencia para '{input_title}'")
    return None

def collaborative_recommender(user_id, ratings, metadata, n_recommendations=10, input_title=None, tfidf_matrix=None):
    start_time = time.time()
    
    if user_id not in ratings['userId'].unique():
        raise ValueError(f"El usuario {user_id} no está en el dataset de ratings.")
    
    # Crear matriz usuario-item
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    user_ids = user_item_matrix.index
    movie_ids = user_item_matrix.columns
    user_index = user_ids.get_loc(user_id)
    
    # Convertir a matriz dispersa para eficiencia
    user_item_sparse = csr_matrix(user_item_matrix.values)
    
    # Encontrar la película de entrada si se proporciona
    input_movie_id = None
    input_movie_title = None
    if input_title:
        # Usamos nuestra función mejorada para encontrar la película
        input_movie_id = find_movie_improved(input_title, metadata)
        
        if input_movie_id is not None:
            movie_row = metadata[metadata['movieId'] == input_movie_id]
            if not movie_row.empty:
                input_movie_title = movie_row['title'].iloc[0]
                print(f"Película de entrada identificada: '{input_title}' -> '{input_movie_title}' (ID: {input_movie_id})")
        else:
            print(f"No se pudo encontrar la película: '{input_title}'")
            # Si no se encuentra la película, seguimos pero sin filtrar ninguna
    
    # Implementar KNN para encontrar usuarios similares
    try:
        k = min(30, len(user_ids) - 1)  # Número de vecinos cercanos
        knn_model = NearestNeighbors(n_neighbors=k+1, metric='cosine')  # +1 porque incluye al propio usuario
        knn_model.fit(user_item_sparse)
        
        distances, indices = knn_model.kneighbors(user_item_sparse[user_index].reshape(1, -1))
        similar_user_indices = indices.flatten()[1:]  # Excluir el propio usuario
        similar_users = [user_ids[idx] for idx in similar_user_indices]
        
        # Filtrar las calificaciones para usar solo las de usuarios similares
        similar_users_ratings = ratings[ratings['userId'].isin(similar_users)]
        
        # Obtener películas que el usuario actual no ha calificado
        user_ratings = user_item_matrix.loc[user_id]
        rated_movie_ids = set(ratings[ratings['userId'] == user_id]['movieId'])
        unrated_movie_ids = [mid for mid in movie_ids if mid not in rated_movie_ids]
        
        # Si hay una película de entrada, asegurarse de que no esté en las recomendaciones
        if input_movie_id is not None and input_movie_id in unrated_movie_ids:
            print(f"Eliminando película de entrada (ID: {input_movie_id}) de la lista de películas no calificadas")
            unrated_movie_ids.remove(input_movie_id)
        
        # Calcular predicciones solo para usuarios similares
        predictions = []
        for movie_id in unrated_movie_ids:
            # Obtener calificaciones para esta película de usuarios similares
            movie_ratings = similar_users_ratings[similar_users_ratings['movieId'] == movie_id]
            
            if len(movie_ratings) == 0:
                continue  # Ningún usuario similar ha calificado esta película
            
            # Calcular las similitudes de usuarios que han calificado esta película
            user_sims = []
            for sim_user in movie_ratings['userId'].unique():
                sim_user_idx = user_ids.get_loc(sim_user)
                sim_score = 1 - distances.flatten()[np.where(indices.flatten() == sim_user_idx)[0][0]]
                user_sims.append((sim_user, sim_score))
            
            # Calcular la calificación predicha usando la similitud como peso
            weighted_sum = 0
            sim_sum = 0
            for sim_user, sim_score in user_sims:
                user_rating = movie_ratings[movie_ratings['userId'] == sim_user]['rating'].values[0]
                weighted_sum += sim_score * user_rating
                sim_sum += sim_score
            
            if sim_sum > 0:
                predicted_rating = weighted_sum / sim_sum
                relevance_percent = ((predicted_rating - 1) / 4) * 100  # Convertir a porcentaje
                predictions.append((movie_id, relevance_percent))
        
        # IMDb-style score: ajustar predicciones con ponderación por popularidad
        m = 1000  # votos mínimos requeridos
        C = metadata['vote_average'].mean()
        scored_predictions = []
        
        for movie_id, predicted_relevance in predictions:
            row = metadata[metadata['movieId'] == movie_id]
            if not row.empty:
                v = row['vote_count'].iloc[0]
                R = row['vote_average'].iloc[0]
                if v > 0:
                    weighted_score = (v / (v + m)) * R + (m / (v + m)) * C
                    final_score = predicted_relevance * (weighted_score / 10)  # Normalizamos a rango 0-100
                    scored_predictions.append((movie_id, final_score))
        
        scored_predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = scored_predictions[:n_recommendations*2]  # Tomamos el doble para tener margen después de filtrar
        
        recommendations = []
        for movie_id, final_score in top_predictions:
            row = metadata[metadata['movieId'] == movie_id]
            if not row.empty:
                title = row['title'].iloc[0]
                
                # Verificar explícitamente que no sea la película de entrada
                if input_movie_id and movie_id == input_movie_id:
                    print(f"Eliminando de recomendaciones: {title} (coincide con ID de búsqueda)")
                    continue
                
                # Verificación adicional por título
                if input_movie_title and title.lower() == input_movie_title.lower():
                    print(f"Eliminando de recomendaciones: {title} (coincide con título exacto)")
                    continue
                    
                vote_avg = row['vote_average'].iloc[0]
                recommendations.append((title, round(vote_avg, 2)))
        
        recommendations = add_diversity(recommendations, metadata, n_recommendations)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        recommendations = recommendations[:n_recommendations]  # Asegurarnos de tener el número correcto de recomendaciones
        
        print(f"Tiempo de ejecución de Collaborative KNN: {time.time() - start_time:.2f} segundos")
        return recommendations
        
    except Exception as e:
        print(f"Error en el recomendador colaborativo: {str(e)}")
        raise e