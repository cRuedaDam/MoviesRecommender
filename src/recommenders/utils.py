import pandas as pd
import random

def find_movie_title(movie_id, metadata):
    """
    Encuentra el título de una película dado su movieId.
    
    Args:
        movie_id (int): ID de la película.
        metadata (pd.DataFrame): DataFrame con metadatos de películas.
    
    Returns:
        str or None: Título de la película o None si no se encuentra.
    """
    movie = metadata[metadata['movieId'] == movie_id]
    return movie['title'].iloc[0] if not movie.empty else None

def add_diversity(recommendations, metadata, n_recommendations):
    """
    Añade diversidad a las recomendaciones seleccionando películas con géneros variados.
    
    Args:
        recommendations (list): Lista de tuplas (título, puntaje).
        metadata (pd.DataFrame): DataFrame con metadatos de películas.
        n_recommendations (int): Número de recomendaciones a devolver.
    
    Returns:
        list: Lista de tuplas (título, puntaje) con diversidad.
    """
    if not recommendations:
        return []
    
    # Crear una lista de géneros para cada recomendación
    rec_with_genres = []
    for title, score in recommendations:
        genres = metadata[metadata['title'] == title]['genres_list'].iloc[0] if title in metadata['title'].values else []
        rec_with_genres.append((title, score, genres))
    
    # Ordenar por puntaje
    rec_with_genres.sort(key=lambda x: x[1], reverse=True)
    
    # Seleccionar recomendaciones diversas
    selected = []
    used_genres = set()
    
    for title, score, genres in rec_with_genres:
        if not selected or any(g not in used_genres for g in genres):
            selected.append((title, score))
            used_genres.update(genres)
        if len(selected) >= n_recommendations:
            break
    
    # Completar con las mejores recomendaciones si es necesario
    for title, score, genres in rec_with_genres:
        if (title, score) not in selected and len(selected) < n_recommendations:
            selected.append((title, score))
    
    # Mezclar para evitar sesgo de orden
    random.shuffle(selected)
    return selected[:n_recommendations]