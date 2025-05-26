from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import time
import re
from src.recommenders.utils import find_movie_title, find_movie_improved, add_diversity
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np

def collaborative_recommender(user_id, ratings, metadata, n_recommendations=10, input_title=None, tfidf_matrix=None):
    """
    Genera recomendaciones colaborativas para un usuario específico 'user_id' 
    usando KNN basado en similitud de usuarios en la matriz de ratings.
    
    Parámetros:
    - ratings: DataFrame con columnas ['userId', 'movieId', 'rating'].
    - metadata: DataFrame con información de películas.
    - n_recommendations: número de recomendaciones finales a devolver.
    - input_title: título de película para referencia (opcional).
    - tfidf_matrix: no usado en esta función, pero incluido en la firma.
    
    Retorna una lista con tuplas (título, voto promedio) de películas recomendadas.
    """
    start_time = time.time()
    
    # Validación de que el usuario existe en los ratings
    if user_id not in ratings['userId'].unique():
        raise ValueError(f"El usuario {user_id} no está en el dataset de ratings.")
    
    # Construcción de la matriz usuario-item con ratings
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    user_ids = user_item_matrix.index
    movie_ids = user_item_matrix.columns
    user_index = user_ids.get_loc(user_id)
    
    # Conversión a matriz dispersa para optimizar cálculo KNN
    user_item_sparse = csr_matrix(user_item_matrix.values)
    
    # Identificación del ID de la película de entrada, si se proporciona
    input_movie_id = None
    input_movie_title = None
    
    if input_title:
        input_movie_id = find_movie_improved(input_title, metadata)
        
        if input_movie_id is not None:
            movie_row = metadata[metadata['movieId'] == input_movie_id]
            if not movie_row.empty:
                input_movie_title = movie_row['title'].iloc[0]
                print(f"Película de entrada identificada: '{input_title}' -> '{input_movie_title}' (ID: {input_movie_id})")
        else:
            print(f"No se pudo encontrar la película: '{input_title}'")
    
    # Creación del modelo KNN para encontrar usuarios similares basados en rating
    try:
        k = min(20, len(user_ids) - 1)  # Número máximo de vecinos similares
        knn_model = NearestNeighbors(n_neighbors=k+1, metric='cosine')  # +1 incluye el propio usuario
        knn_model.fit(user_item_sparse)

        print(f"El numero (K) de vecinos definido es {k}")
        
        # Obtener índices y distancias de vecinos más cercanos
        distances, indices = knn_model.kneighbors(user_item_sparse[user_index].reshape(1, -1))
        similar_user_indices = indices.flatten()[1:]  # Excluye el propio usuario
        similar_users = [user_ids[idx] for idx in similar_user_indices]
        
        # Filtrar ratings para incluir solo los usuarios similares
        similar_users_ratings = ratings[ratings['userId'].isin(similar_users)]
        
        # Películas no calificadas por el usuario objetivo
        user_ratings = user_item_matrix.loc[user_id]
        rated_movie_ids = set(ratings[ratings['userId'] == user_id]['movieId'])
        unrated_movie_ids = [mid for mid in movie_ids if mid not in rated_movie_ids]
        
        # Eliminar la película de entrada de la lista de no calificadas si está presente
        if input_movie_id is not None and input_movie_id in unrated_movie_ids:
            print(f"Eliminando película de entrada (ID: {input_movie_id}) de la lista de películas no calificadas")
            unrated_movie_ids.remove(input_movie_id)
        
        # Calcular predicciones de rating para películas no calificadas
        predictions = []
        for movie_id in unrated_movie_ids:
            movie_ratings = similar_users_ratings[similar_users_ratings['movieId'] == movie_id]
            
            if len(movie_ratings) == 0:
                continue  # Ningún usuario similar calificó esta película
            
            # Calcular similitud para usuarios que calificaron esta película
            user_sims = []
            for sim_user in movie_ratings['userId'].unique():
                sim_user_idx = user_ids.get_loc(sim_user)
                sim_score = 1 - distances.flatten()[np.where(indices.flatten() == sim_user_idx)[0][0]]
                user_sims.append((sim_user, sim_score))
            
            # Calcular rating predicho usando las similitudes como pesos
            weighted_sum = 0
            sim_sum = 0
            for sim_user, sim_score in user_sims:
                user_rating = movie_ratings[movie_ratings['userId'] == sim_user]['rating'].values[0]
                weighted_sum += sim_score * user_rating
                sim_sum += sim_score
            
            if sim_sum > 0:
                predicted_rating = weighted_sum / sim_sum
                # Convertir la predicción a porcentaje de relevancia
                relevance_percent = ((predicted_rating - 1) / 4) * 100
                predictions.append((movie_id, relevance_percent))
        
        # Ajustar predicciones con una puntuación ponderada basada en popularidad (IMDb-style)
        m = 1000  # votos mínimos para confiabilidad
        C = metadata['vote_average'].mean()  # promedio general de votos
        scored_predictions = []
        
        for movie_id, predicted_relevance in predictions:
            row = metadata[metadata['movieId'] == movie_id]
            if not row.empty:
                v = row['vote_count'].iloc[0]
                R = row['vote_average'].iloc[0]
                if v > 0:
                    weighted_score = (v / (v + m)) * R + (m / (v + m)) * C
                    final_score = predicted_relevance * (weighted_score / 10)
                    scored_predictions.append((movie_id, final_score))
        
        # Ordenar las predicciones por la puntuación final de mayor a menor
        scored_predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = scored_predictions[:n_recommendations*2]  # Tomar más para filtrar después
        
        recommendations = []
        for movie_id, final_score in top_predictions:
            row = metadata[metadata['movieId'] == movie_id]
            if not row.empty:
                title = row['title'].iloc[0]
                
                # Excluir explícitamente la película de entrada por ID o título
                if input_movie_id and movie_id == input_movie_id:
                    print(f"Eliminando de recomendaciones: {title} (coincide con ID de búsqueda)")
                    continue
                
                if input_movie_title and title.lower() == input_movie_title.lower():
                    print(f"Eliminando de recomendaciones: {title} (coincide con título exacto)")
                    continue
                    
                vote_avg = row['vote_average'].iloc[0]
                recommendations.append((title, round(vote_avg, 2)))
        
        # Añadir diversidad a las recomendaciones
        recommendations = add_diversity(recommendations, metadata, n_recommendations)
        # Ordenar y recortar a la cantidad solicitada
        recommendations.sort(key=lambda x: x[1], reverse=True)
        recommendations = recommendations[:n_recommendations]
        
        print(f"Tiempo de ejecución de Collaborative KNN: {time.time() - start_time:.2f} segundos")
        return recommendations
        
    except Exception as e:
        print(f"Error en el recomendador colaborativo: {str(e)}")
        raise e

def evaluate_model(model_fn, ratings, metadata, k=10, test_size=0.2, n_users=100):
    """
    Evalúa un sistema de recomendación con métricas top-k.

    Parámetros:
    - model_fn: función recomendadora, debe aceptar (user_id, ratings, metadata, n_recommendations).
    - ratings: DataFrame original con ratings.
    - metadata: DataFrame con metadata de películas.
    - k: número de recomendaciones a considerar.
    - test_size: proporción de ratings a usar como test.
    - n_users: número de usuarios aleatorios a evaluar.

    Retorna:
    - Diccionario con métricas promedio: Precision@k, Recall@k y NDCG@k.
    """
    # Dividir los datos en entrenamiento y prueba
    train_data, test_data = train_test_split(ratings, test_size=test_size, random_state=42)
    
    # Construir conjuntos de prueba por usuario
    test_ratings_by_user = test_data.groupby('userId')['movieId'].apply(set).to_dict()
    
    # Evaluar sobre una muestra de usuarios
    user_sample = np.random.choice(list(test_ratings_by_user.keys()), size=min(n_users, len(test_ratings_by_user)), replace=False)

    precision_list = []
    recall_list = []
    ndcg_list = []

    for user_id in user_sample:
        relevant_items = test_ratings_by_user[user_id]
        if not relevant_items:
            continue  # saltar si no hay test ratings
        
        try:
            recommendations = model_fn(user_id, train_data, metadata, n_recommendations=k)
            recommended_titles = [title for title, _ in recommendations]
            
            # Mapear títulos recomendados a IDs
            recommended_ids = metadata[metadata['title'].isin(recommended_titles)]['movieId'].values
            
            hits = len(set(recommended_ids) & relevant_items)
            precision = hits / k
            recall = hits / len(relevant_items)
            
            # NDCG
            dcg = 0.0
            idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k))])
            for i, rec_id in enumerate(recommended_ids):
                if rec_id in relevant_items:
                    dcg += 1.0 / np.log2(i + 2)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            
            precision_list.append(precision)
            recall_list.append(recall)
            ndcg_list.append(ndcg)

        except Exception as e:
            print(f"Error al evaluar usuario {user_id}: {e}")
            continue

    return {
        'Precision@K': np.mean(precision_list),
        'Recall@K': np.mean(recall_list),
        'NDCG@K': np.mean(ndcg_list)
    }
