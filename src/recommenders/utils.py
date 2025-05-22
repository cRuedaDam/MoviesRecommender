import requests
import time
from ratelimit import limits, sleep_and_retry
from dotenv import load_dotenv
import os
import random

# Carga las variables de entorno desde un archivo .env
load_dotenv()

# Configuración de la API de TMDB
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "a5a6289ec4bde4a1fcba8f6d49580f71")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://www.themoviedb.org/t/p/"

# Límite de tasa: 50 peticiones por segundo (con control automático)
CALLS = 50
PERIOD = 1  # segundos

@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
def get_poster_url(tmdb_id, width=342):
    """
    Devuelve la URL del póster de una película usando el ID de TMDB.
    Si el ID es inválido o no hay imagen disponible, se devuelve una imagen placeholder.

    Args:
        tmdb_id (int | str | float): ID de TMDB.
        width (int): Ancho de la imagen (por defecto 342px).

    Returns:
        str: URL del póster o imagen por defecto si no se encuentra.
    """
    placeholder = "https://via.placeholder.com/300x450?text=Poster+no+disponible"

    print(f"Validating tmdbId: {tmdb_id} (type: {type(tmdb_id)})")

    # Validación del ID
    try:
        tmdb_id_clean = int(float(tmdb_id)) if isinstance(tmdb_id, (str, float)) else int(tmdb_id)
        if tmdb_id_clean <= 0:
            print(f"Invalid tmdbId: {tmdb_id_clean} (non-positive), returning placeholder")
            return placeholder
    except (ValueError, TypeError) as e:
        print(f"Invalid tmdbId: {tmdb_id} (conversion error: {e}), returning placeholder")
        return placeholder

    # Construcción de la URL para la API de TMDB
    api_url = f"{TMDB_BASE_URL}/movie/{tmdb_id_clean}?api_key={TMDB_API_KEY}&language=en-US"

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()  # Lanza error si el status code no es 200

        data = response.json()
        poster_path = data.get('poster_path')

        # Verificación del campo 'poster_path'
        if not poster_path or not isinstance(poster_path, str) or not poster_path.endswith(('.jpg', '.jpeg', '.png')):
            print(f"No valid poster_path for tmdbId {tmdb_id_clean}, received: {poster_path}")
            return placeholder

        full_url = f"{TMDB_IMAGE_BASE_URL}w{width}{poster_path}"
        print(f"Successfully fetched poster for tmdbId {tmdb_id_clean}: {full_url}")
        return full_url

    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching poster for tmdb_id {tmdb_id_clean}: {e}")
        return placeholder


def find_movie_title(movie_id, metadata):
    """
    Localiza el título de una película usando su movieId en el DataFrame metadata.

    Args:
        movie_id (int): ID interno de la película.
        metadata (pd.DataFrame): DataFrame con información de películas.

    Returns:
        str | None: Título de la película si se encuentra, o None.
    """
    movie = metadata[metadata['movieId'] == movie_id]
    return movie['title'].iloc[0] if not movie.empty else None


def add_diversity(recommendations, metadata, n_recommendations):
    """
    Aplica diversidad a una lista de recomendaciones basada en géneros diferentes.

    Args:
        recommendations (list): Lista de tuplas (título, score).
        metadata (pd.DataFrame): DataFrame con información de películas.
        n_recommendations (int): Número de recomendaciones finales deseadas.

    Returns:
        list: Lista diversificada de recomendaciones (título, score).
    """
    if not recommendations:
        return []

    rec_with_genres = []

    # Asocia cada título con sus géneros
    for title, score in recommendations:
        if title in metadata['title'].values:
            genres = metadata[metadata['title'] == title]['genres_list'].iloc[0]
        else:
            genres = []
        rec_with_genres.append((title, score, genres))

    # Ordena por score descendente
    rec_with_genres.sort(key=lambda x: x[1], reverse=True)

    selected = []
    used_genres = set()

    # Selecciona recomendaciones que agregan nuevos géneros
    for title, score, genres in rec_with_genres:
        if not selected or any(g not in used_genres for g in genres):
            selected.append((title, score))
            used_genres.update(genres)
        if len(selected) >= n_recommendations:
            break

    # Rellena con títulos restantes si no se alcanza el número requerido
    for title, score, genres in rec_with_genres:
        if (title, score) not in selected and len(selected) < n_recommendations:
            selected.append((title, score))

    # Mezcla el resultado para evitar un orden predecible
    random.shuffle(selected)
    return selected[:n_recommendations]
