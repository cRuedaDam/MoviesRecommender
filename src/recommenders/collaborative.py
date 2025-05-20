from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import time
from src.recommenders.utils import find_movie_title, add_diversity

def collaborative_recommender(user_id, ratings, metadata, n_recommendations=10, input_title=None, tfidf_matrix=None):
    start_time = time.time()

    if user_id not in ratings['userId'].unique():
        raise ValueError(f"El usuario {user_id} no está en el dataset de ratings.")

    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    user_ids = user_item_matrix.index
    movie_ids = user_item_matrix.columns
    user_index = user_ids.get_loc(user_id)

    user_item_sparse = csr_matrix(user_item_matrix.values)
    user_similarity = cosine_similarity(user_item_sparse)
    sim_scores = user_similarity[user_index]

    user_ratings = user_item_matrix.loc[user_id]
    unrated_movie_ids = user_ratings[user_ratings == 0].index

    predictions = []

    for movie_id in unrated_movie_ids:
        users_who_rated = user_item_matrix[user_item_matrix[movie_id] > 0].index

        sims = []
        ratings_for_movie = []
        for other_user in users_who_rated:
            idx = user_ids.get_loc(other_user)
            sims.append(sim_scores[idx])
            ratings_for_movie.append(user_item_matrix.at[other_user, movie_id])

        sims = np.array(sims)
        ratings_for_movie = np.array(ratings_for_movie)

        mask = sims > 0
        sims = sims[mask]
        ratings_for_movie = ratings_for_movie[mask]

        if sims.size == 0:
            continue

        weighted_sum = np.sum(sims * ratings_for_movie)
        sum_sims = np.sum(sims)

        predicted_rating = weighted_sum / sum_sims
        relevance_percent = ((predicted_rating - 1) / 4) * 100

        predictions.append((movie_id, relevance_percent))

    # IMDb-style score: ajustar predicciones con ponderación por popularidad
    m = 100  # votos mínimos requeridos
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
    top_predictions = scored_predictions[:n_recommendations]

    recommendations = []
    rated_movie_ids = set(ratings[ratings['userId'] == user_id]['movieId'])  # Películas ya vistas

    for movie_id, final_score in top_predictions:
        if movie_id in rated_movie_ids:
            continue  # Saltar películas ya vistas

        row = metadata[metadata['movieId'] == movie_id]
        if not row.empty:
            title = row['title'].iloc[0]
            vote_avg = row['vote_average'].iloc[0]
            recommendations.append((title, round(vote_avg, 2)))

    recommendations = add_diversity(recommendations, metadata, n_recommendations)
    recommendations.sort(key=lambda x: x[1], reverse=True)

    print(f"Tiempo de ejecución de Collaborative: {time.time() - start_time:.2f} segundos")
    return recommendations
