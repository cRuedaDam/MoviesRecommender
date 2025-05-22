from typing import List, Tuple, Dict
import time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler
from src.recommenders.utils import find_movie_title

# Pesos de los distintos 
DEFAULT_WEIGHTS: Dict[str, float] = {
    'similarity':   0.30,
    'collection':   0.15,
    'genre':        0.10,
    'main_genre':   0.15,
    'year':         0.05,
    'language':     0.05,
    'popularity':   0.10,
    'rating':       0.05,
}

def rescale_similarities_minmax(raw_sim: np.ndarray) -> np.ndarray:
    """
    MinMaxScaler: reescala coseno a [0,1].
    """
    sims = raw_sim.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(sims).flatten()

def validate_title(input_title: str, metadata: pd.DataFrame) -> str:
    if not isinstance(input_title, str) or not input_title.strip():
        raise ValueError("El título de entrada debe ser una cadena no vacía.")
    if input_title not in metadata['title'].values:
        suggestion = find_movie_title(input_title, metadata['title'].values)
        if suggestion:
            print(f"No se encontró '{input_title}'. Quizás quisiste decir '{suggestion}'.")
            return suggestion
        raise ValueError(f"No se puede generar recomendaciones para '{input_title}'.")
    return input_title

def compute_year_proximity(
    years: pd.Series,
    target_year: int,
    max_diff: int = 10
) -> np.ndarray:
    diffs = (years - target_year).abs().clip(upper=max_diff)
    scores = 1 - (diffs / max_diff)
    return scores.fillna(0.2).values

def content_based_recommender(
    input_title: str,
    metadata: pd.DataFrame,
    n_recommendations: int = 10,
    tfidf_matrix=None,
    weights: Dict[str, float] = None
) -> List[Tuple[str, float]]:
    """
    Genera recomendaciones basadas en contenido para una película dada,
    usando reescalado MinMax para la similitud y normalizando todos los factores en [0,1].
    Devuelve la lista de (título, score%) con score entre 0 y 100.
    """
    start_time = time.time()
    weights = weights or DEFAULT_WEIGHTS

    # 1. Validar título
    title = validate_title(input_title, metadata)

    # 2. Extraer metadatos de la película de entrada
    movie_idx = metadata.index[metadata['title'] == title][0]
    meta = metadata.loc[movie_idx]
    movie_genres = set(meta['genres_list'])
    main_genre = next(iter(movie_genres), None)

    print(f"Película de entrada: {title} ({meta.release_year}) — Géneros: {', '.join(movie_genres)}")

    # 3. Similitud coseno cruda
    raw_sim = linear_kernel(tfidf_matrix[movie_idx:movie_idx+1], tfidf_matrix).flatten()

    # 4. Reescalado MinMax de similitud a [0,1]
    similarity = rescale_similarities_minmax(raw_sim)

    # 5. Construcción de DataFrame con factores
    df = metadata.copy().reset_index(drop=True)
    df['similarity'] = similarity
    df = df[df['title'] != title].reset_index(drop=True)

    # Factores adicionales
    df['collection'] = (
        (df['belongs_to_collection'] == meta['belongs_to_collection'])
        .astype(float)
    )
    df['genre'] = df['genres_list'].apply(
        lambda x: len(set(x) & movie_genres) / max(len(movie_genres), 1)
    )
    df['main_genre'] = df['genres_list'].apply(
        lambda x: 1.0 if main_genre in x else 0.0
    )
    df['year'] = compute_year_proximity(df['release_year'], meta['release_year'])
    df['language'] = df['original_language'].apply(
        lambda x: 1.0 if x == meta['original_language'] else 0.2
    )
    max_votes = df['vote_count'].max()
    df['popularity'] = df['vote_count'].apply(
        lambda x: 0.2 + 0.8 * (np.log1p(x) / np.log1p(max_votes))
    )
    df['rating'] = df['vote_average'].apply(
        lambda x: 0.2 + 0.8 * (x / 10.0)
    )

    # 6. Normalizar cualquier desviación en [0,1]
    norm_cols = [
        'similarity','collection','genre','main_genre',
        'year','language','popularity','rating'
    ]
    for col in norm_cols:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            df[col] = (df[col] - col_min) / (col_max - col_min)

    # 7. Score combinado en [0,1]
    df['combined_score'] = sum(df[col] * weights[col] for col in norm_cols)

    # 8. Convertir a porcentaje 0–100
    df['combined_score'] = df['combined_score'] * 100

    # 9. Selección y ordenamiento
    top = df.nlargest(n_recommendations, 'combined_score')[['title','combined_score']]

    print(f"Tiempo de ejecución: {time.time() - start_time:.2f} segundos")
    # Devolvemos lista de (título, score_porcentaje)
    return list(top.itertuples(index=False, name=None))
