import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import ast
import time

# --------------------------------------
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# --------------------------------------
def load_and_preprocess_data(data_folder='dataset'):
    """
    Carga, limpia y preprocesa los datos de películas y ratings.

    Returns:
        tuple: (metadata, ratings, tfidf_matrix)
            - metadata (pd.DataFrame): Datos de las películas con columnas relevantes.
            - ratings (pd.DataFrame): Datos de los ratings de los usuarios.
            - tfidf_matrix (csr_matrix): Matriz TF-IDF para descripciones de películas.
    """
    start_time = time.time()
    try:
        metadata = pd.read_csv(f'{data_folder}/movies_metadata.csv', low_memory=False)
        ratings = pd.read_csv(f'{data_folder}/ratings_small.csv')
        links = pd.read_csv(f'{data_folder}/links.csv')
    except FileNotFoundError as e:
        print(f"Error al cargar los archivos: {e}")
        return None, None, None

    metadata = metadata[metadata['id'].apply(lambda x: str(x).isdigit())].copy()
    metadata['id'] = metadata['id'].astype(int)

    metadata['title'] = metadata['title'].astype(str).str.strip()
    metadata['vote_average'] = pd.to_numeric(metadata['vote_average'], errors='coerce')
    metadata['vote_count'] = pd.to_numeric(metadata['vote_count'], errors='coerce')
    metadata['runtime'] = pd.to_numeric(metadata['runtime'], errors='coerce')
    metadata['release_date'] = pd.to_datetime(metadata['release_date'], errors='coerce')
    metadata['release_year'] = metadata['release_date'].dt.year
    metadata['imdb_id'] = metadata['imdb_id'].astype(str)

    links = links[links['movieId'].notna()].copy()
    links['movieId'] = links['movieId'].astype(int)
    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce').fillna(0).astype(int)

    metadata_ids = set(metadata['id'])
    links_ids = set(links['tmdbId'])
    common_ids = metadata_ids.intersection(links_ids)
    print(f"Número de IDs en común: {len(common_ids)}")

    metadata = metadata[metadata['id'].isin(common_ids)]
    metadata = metadata.merge(links, left_on='id', right_on='tmdbId', how='inner')

    def parse_genres(genres_str):
        if pd.isna(genres_str) or genres_str is None:
            return "", []
        try:
            genres = ast.literal_eval(str(genres_str))
            if isinstance(genres, list):
                genre_names = [g['name'] for g in genres if isinstance(g, dict) and 'name' in g]
                return " ".join(genre_names), genre_names
            return "", []
        except (ValueError, SyntaxError, TypeError) as e:
            print(f"Error al procesar géneros: {e}, valor: {genres_str}")
            return "", []

    metadata[['genres_clean', 'genres_list']] = pd.DataFrame(metadata['genres'].apply(parse_genres).tolist(), index=metadata.index)

    metadata['overview'] = metadata['overview'].fillna('')
    metadata['description'] = metadata['overview'] + ' ' + metadata['genres_clean']

    metadata['vote_count_norm'] = metadata['vote_count'].apply(lambda x: x / metadata['vote_count'].max() if pd.notna(x) else 0)
    metadata['vote_score'] = metadata.apply(
        lambda x: (x['vote_average'] * x['vote_count_norm']) if pd.notna(x['vote_average']) else 0, 
        axis=1
    )

    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    tfidf_matrix = tfidf.fit_transform(metadata['overview'].fillna(''))

    metadata = metadata[[
        'id', 'movieId', 'imdb_id', 'title', 'genres', 'genres_clean', 'genres_list', 'overview',
        'description', 'belongs_to_collection', 'poster_path',
        'vote_average', 'vote_count', 'vote_count_norm', 'vote_score', 'runtime', 'release_date', 'release_year',
        'original_language'
    ]]
    
    end_time = time.time()
    print(f"Tiempo de carga y preprocesamiento: {end_time - start_time:.2f} segundos")
    return metadata, ratings, tfidf_matrix

# --------------------------------------
# 2. FUNCIONES DE UTILIDAD
# --------------------------------------
def find_movie_title(title, metadata):
    """
    Busca un título de película en el dataset. Si no hay coincidencia exacta, busca coincidencias parciales.

    Args:
        title (str): Título de la película a buscar.
        metadata (pd.DataFrame): Datos de las películas.

    Returns:
        str: Título encontrado (exacto o más relevante), o None si no hay coincidencias.
    """
    title = title.strip()
    if title in metadata['title'].values:
        return title
    
    matches = metadata[metadata['title'].str.contains(title, case=False, na=False)]
    if not matches.empty:
        matches = matches.sort_values(by='vote_score', ascending=False)
        selected_title = matches['title'].iloc[0]
        print(f"No se encontró '{title}' exactamente. Usando '{selected_title}' como la mejor coincidencia.")
        return selected_title
    
    suggestions = metadata[metadata['title'].str.contains(title.split()[0], case=False, na=False)]['title'].head(5).tolist()
    if suggestions:
        print(f"No se encontró '{title}'. Títulos sugeridos: {', '.join(suggestions)}")
    else:
        print(f"No se encontró '{title}' ni coincidencias parciales en el dataset.")
    return None

def add_diversity(recommendations, metadata, max_same_collection=2, max_similar_genres=3):
    """
    Re-ordena las recomendaciones para aumentar la diversidad, respetando el orden de similitud.

    Args:
        recommendations (list): Lista de tuplas (título, similitud).
        metadata (pd.DataFrame): Datos de las películas.
        max_same_collection (int): Máximo de películas de la misma colección.
        max_similar_genres (int): Máximo de películas con géneros similares.

    Returns:
        list: Lista de tuplas (título, similitud) con mayor diversidad, ordenada por similitud.
    """
    diverse_recs = []
    collection_count = {}
    genre_count = {}

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

    for title, sim in recommendations:
        if title not in metadata['title'].values:
            diverse_recs.append((title, sim))
            continue

        idx = metadata[metadata['title'] == title].index[0]
        collection = metadata.loc[idx, 'belongs_to_collection']
        genres = set(metadata.loc[idx, 'genres_list'])

        collection_id = None
        if pd.notna(collection):
            try:
                collection_id = ast.literal_eval(collection).get('id', 'unknown') if isinstance(collection, str) else collection.get('id', 'unknown')
            except:
                collection_id = 'unknown'

        genre_key = "-".join(sorted(list(genres))[:2])

        should_include = True
        if collection_id is not None and collection_count.get(collection_id, 0) >= max_same_collection:
            should_include = False
        if genre_count.get(genre_key, 0) >= max_similar_genres:
            should_include = False

        if should_include:
            diverse_recs.append((title, sim))
            if collection_id is not None:
                collection_count[collection_id] = collection_count.get(collection_id, 0) + 1
            genre_count[genre_key] = genre_count.get(genre_key, 0) + 1

    if len(diverse_recs) < len(recommendations):
        remaining = [r for r in recommendations if r[0] not in [d[0] for d in diverse_recs]]
        diverse_recs.extend(remaining[:len(recommendations) - len(diverse_recs)])

    diverse_recs = sorted(diverse_recs, key=lambda x: x[1], reverse=True)
    return diverse_recs[:len(recommendations)]

# --------------------------------------
# 3. FILTRADO BASADO EN CONTENIDO
# --------------------------------------
def content_based_recommender(title, metadata, n_recommendations, tfidf_matrix, genre_weight=0.6):
    """
    Genera recomendaciones basadas en el contenido de una película dada.

    Args:
        title (str): Título de la película de entrada.
        metadata (pd.DataFrame): Datos de las películas.
        n_recommendations (int): Número de recomendaciones a generar.
        tfidf_matrix (csr_matrix): Matriz TF-IDF para similitud de texto.
        genre_weight (float): Ponderación para la similitud de géneros.

    Returns:
        list: Lista de tuplas (título, similitud) de las películas recomendadas.

    Raises:
        ValueError: Si la película no se encuentra en el dataset.
    """
    start_time = time.time()
    title = find_movie_title(title, metadata)
    if title is None:
        raise ValueError(f"No se puede generar recomendaciones para '{title}' porque no se encontró en el dataset.")

    idx = metadata[metadata['title'] == title].index[0]
    input_genres = set(metadata.loc[idx, 'genres_list'])
    input_language = metadata.loc[idx, 'original_language']
    input_year = metadata.loc[idx, 'release_year']
    input_collection = metadata.loc[idx, 'belongs_to_collection']

    print(f"Película de entrada: {title}, Idioma: {input_language}, Año: {input_year}, Géneros: {input_genres}, Colección: {input_collection}")

    related_films = metadata[metadata['title'].str.contains(title.split()[0], case=False, na=False)][
        ['title', 'movieId', 'release_year', 'genres_list', 'belongs_to_collection', 'vote_count']
    ]
    print(f"\nPelículas relacionadas con '{title}' en el dataset:")
    print(related_films)

    if pd.notna(input_collection):
        try:
            # Parsear la colección de entrada
            input_collection_dict = ast.literal_eval(input_collection) if isinstance(input_collection, str) else input_collection
            collection_id = input_collection_dict.get('id') if isinstance(input_collection_dict, dict) else None
            
            if collection_id:
                def is_same_collection(collection):
                    if pd.isna(collection):
                        return False
                    try:
                        # Parsear la colección de cada película
                        collection_dict = ast.literal_eval(collection) if isinstance(collection, str) else collection
                        if isinstance(collection_dict, dict):
                            movie_collection_id = collection_dict.get('id')
                            return movie_collection_id == collection_id
                        return False
                    except (ValueError, SyntaxError, TypeError) as e:
                        return False

                collection_films = metadata[metadata['belongs_to_collection'].apply(is_same_collection)].copy()
                
                # Filtrar por géneros compatibles
                collection_films = collection_films[collection_films['genres_list'].apply(
                    lambda x: len(set(x) & input_genres) >= 1 and 'Horror' not in x and 'Science Fiction' not in x
                )][['title', 'movieId', 'release_year', 'genres_list', 'vote_count']]
                
                if not collection_films.empty:
                    print(f"\nPelículas en la misma colección:")
                    print(collection_films)
                else:
                    print("\nNo se encontraron películas en la misma colección con géneros compatibles.")
        except (ValueError, SyntaxError, TypeError) as e:
            print(f"No se pudo extraer información de colección: {e}")

    cosine_sim = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()

    def genre_similarity(genres_list):
        if not genres_list or not input_genres:
            return 0.0
        target_genres = set(genres_list)
        intersection = len(input_genres & target_genres)
        union = len(input_genres | target_genres)
        return intersection / union if union > 0 else 0.0

    genre_sim = metadata['genres_list'].apply(genre_similarity).values

    collection_sim = np.zeros(len(metadata))
    if pd.notna(input_collection):
        try:
            input_collection_dict = ast.literal_eval(input_collection) if isinstance(input_collection, str) else input_collection
            input_collection_id = input_collection_dict.get('id') if isinstance(input_collection_dict, dict) else None
            if input_collection_id:
                for i, collection in enumerate(metadata['belongs_to_collection']):
                    if pd.notna(collection):
                        try:
                            collection_dict = ast.literal_eval(collection) if isinstance(collection, str) else collection
                            collection_id = collection_dict.get('id') if isinstance(collection_dict, dict) else None
                            if collection_id == input_collection_id:
                                collection_sim[i] = 1.0
                        except:
                            pass
        except (ValueError, SyntaxError, TypeError):
            print("Error al procesar colecciones")

    popularity_sim = metadata['vote_score'].values

    combined_sim = (
        genre_weight * genre_sim +
        0.15 * collection_sim +
        0.05 * popularity_sim +
        0.1 * cosine_sim
    )
    combined_sim = combined_sim / combined_sim.max() * 100
    year_sim = metadata['release_year'].apply(
        lambda x: 1.0 if pd.notna(x) and pd.notna(input_year) and abs(x - input_year) <= 10 else 0.9
    ).values
    combined_sim *= year_sim

    valid_indices = metadata[
        ((metadata['original_language'] == input_language) | (metadata['vote_count'] >= 200)) &
        (metadata['vote_count'] >= 50) &
        (metadata['genres_list'].apply(lambda x: len(set(x) & input_genres) >= 1))
    ].index

    if len(valid_indices) < n_recommendations * 2:
        print(f"Incluyendo películas en otros idiomas para tener suficientes candidatos")
        other_indices = metadata[
            (metadata['original_language'] != input_language) &
            (metadata['vote_count'] >= 50) &
            (metadata['genres_list'].apply(lambda x: len(set(x) & input_genres) >= 1))
        ].index
        valid_indices = valid_indices.union(other_indices)

    sim_scores = [s for s in list(enumerate(combined_sim)) if s[0] in valid_indices]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:n_recommendations * 2]

    movie_indices = [i for i, _ in sim_scores]
    similarities = [score for _, score in sim_scores]
    titles = metadata['title'].iloc[movie_indices].tolist()
    result = list(zip(titles, similarities))
    result = add_diversity(result, metadata, max_same_collection=2, max_similar_genres=3)[:n_recommendations]
    
    end_time = time.time()
    print(f"Tiempo de ejecución de Content Based: {end_time - start_time:.2f} segundos")
    return result

# --------------------------------------
# 4. FILTRADO COLABORATIVO
# --------------------------------------
def collaborative_recommender(user_id, ratings, metadata, n_recommendations, input_title=None, tfidf_matrix=None):
    """
    Genera recomendaciones colaborativas para un usuario dado, basadas en una película de entrada.

    Args:
        user_id (int): ID del usuario.
        ratings (pd.DataFrame): Datos de los ratings de los usuarios.
        metadata (pd.DataFrame): Datos de las películas.
        n_recommendations (int): Número de recomendaciones a generar.
        input_title (str, optional): Título de la película para filtrar por géneros.
        tfidf_matrix (csr_matrix, optional): Matriz TF-IDF para similitud de texto.

    Returns:
        list: Lista de tuplas (título, similitud) de las películas recomendadas.
    """
    start_time = time.time()
    if user_id not in ratings['userId'].values:
        raise ValueError(f"El usuario con ID {user_id} no se encuentra en el dataset.")

    input_genres = []
    input_idx = None
    input_collection_id = None
    if input_title:
        input_title = find_movie_title(input_title, metadata)
        if input_title is None:
            print(f"No se puede filtrar por géneros porque '{input_title}' no se encontró. Usando recomendaciones generales.")
        else:
            idx = metadata[metadata['title'] == input_title].index[0]
            input_genres = metadata.loc[idx, 'genres_list']
            input_idx = idx
            if pd.notna(metadata.loc[idx, 'belongs_to_collection']):
                try:
                    collection_dict = ast.literal_eval(metadata.loc[idx, 'belongs_to_collection']) if isinstance(metadata.loc[idx, 'belongs_to_collection'], str) else metadata.loc[idx, 'belongs_to_collection']
                    input_collection_id = collection_dict.get('id') if isinstance(collection_dict, dict) else None
                except:
                    pass

    min_ratings = 10
    min_user_ratings = 10
    rating_counts = ratings['movieId'].value_counts()
    valid_movie_ids = rating_counts[rating_counts >= min_ratings].index
    user_counts = ratings['userId'].value_counts()
    valid_user_ids = user_counts[user_counts >= min_user_ratings].index
    ratings_filtered = ratings[
        (ratings['movieId'].isin(valid_movie_ids)) &
        (ratings['userId'].isin(valid_user_ids))
    ].copy()

    print(f"\nColaborativo - Películas tras filtrado: {len(valid_movie_ids)}")
    print(f"Colaborativo - Usuarios tras filtrado: {len(valid_user_ids)}")

    user_movie_matrix = ratings_filtered.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    user_movie_matrix_sparse = csr_matrix(user_movie_matrix.values)

    user_similarity = cosine_similarity(user_movie_matrix_sparse)

    if user_id not in user_movie_matrix.index:
        print(f"El usuario {user_id} no tiene suficientes valoraciones para el filtrado colaborativo.")
        popular_movies = metadata.sort_values(by='vote_score', ascending=False)[:n_recommendations * 2]
        if input_genres:
            popular_movies = popular_movies[
                popular_movies['genres_list'].apply(lambda x: len(set(x) & set(input_genres)) >= 3)
            ]
        popular_titles = popular_movies['title'].tolist()[:n_recommendations]
        popular_scores = [(popular_movies['vote_average'].iloc[i] / 10) * 100 for i in range(len(popular_titles))]
        return list(zip(popular_titles, popular_scores))

    user_idx = user_movie_matrix.index.get_loc(user_id)
    similar_users_indices = np.argsort(user_similarity[user_idx])[::-1][1:100]
    similar_users_similarity = user_similarity[user_idx][similar_users_indices]
    similar_users_ids = user_movie_matrix.index[similar_users_indices]

    user_ratings = user_movie_matrix.loc[user_id]
    avg_user_rating = ratings_filtered[ratings_filtered['userId'] == user_id]['rating'].mean()

    ranked_movies = {}
    sum_similarity = {}
    for i, similar_user_id in enumerate(similar_users_ids):
        similarity_score = similar_users_similarity[i]
        if similarity_score > 0:
            similar_user_rated_movies = user_movie_matrix.loc[similar_user_id]
            similar_user_avg = ratings_filtered[ratings_filtered['userId'] == similar_user_id]['rating'].mean()
            for movie_id, rating in similar_user_rated_movies.items():
                if user_ratings.get(movie_id, 0) == 0 and rating > 0:
                    adjusted_rating = rating - similar_user_avg + avg_user_rating
                    ranked_movies[movie_id] = ranked_movies.get(movie_id, 0) + adjusted_rating * similarity_score
                    sum_similarity[movie_id] = sum_similarity.get(movie_id, 0) + similarity_score

    normalized_scores = {}
    avg_ratings = {}
    for movie_id in ranked_movies.keys():
        avg_ratings[movie_id] = ratings_filtered[ratings_filtered['movieId'] == movie_id]['rating'].mean()

    for movie_id, total_weighted_rating in ranked_movies.items():
        if movie_id in sum_similarity and sum_similarity[movie_id] > 0:
            weighted_rating = total_weighted_rating / sum_similarity[movie_id]
            movie_avg = avg_ratings.get(movie_id, 0)
            normalized_scores[movie_id] = (
                0.7 * (weighted_rating / 5.0) +
                0.3 * (movie_avg / 5.0)
            )
        else:
            normalized_scores[movie_id] = 0.0

    max_score = max(normalized_scores.values(), default=1)
    if max_score > 0:
        normalized_scores = {k: (v / max_score) * 100 for k, v in normalized_scores.items()}

    top_movie_ids = sorted(normalized_scores, key=normalized_scores.get, reverse=True)[:n_recommendations * 20]

    recommended = metadata[metadata['movieId'].isin(top_movie_ids)][['movieId', 'title', 'genres_list', 'belongs_to_collection']].copy()
    if input_genres:
        def genre_overlap(genres_list):
            if not genres_list or not input_genres:
                return 0.0
            target_genres = set(genres_list)
            input_genres_set = set(input_genres)
            intersection = len(input_genres_set & target_genres)
            return intersection / len(input_genres_set) if len(input_genres_set) > 0 else 0.0

        recommended['genre_score'] = recommended['genres_list'].apply(genre_overlap)
        recommended = recommended[recommended['genre_score'] >= 0.66]  # Requerir los 3 géneros

    recommended['similarity'] = recommended['movieId'].map(normalized_scores)
    recommended = recommended.sort_values(by='similarity', ascending=False).head(n_recommendations * 2)

    if len(recommended) < n_recommendations and input_idx is not None:
        print(f"Advertencia: Solo se encontraron {len(recommended)} recomendaciones colaborativas.")
        needed = n_recommendations - len(recommended)
        # Priorizar películas de la misma colección
        collection_films = pd.DataFrame()
        if input_collection_id:
            collection_films = metadata[metadata['belongs_to_collection'].apply(
                lambda x: pd.notna(x) and isinstance(x, (str, dict)) and (
                    (isinstance(x, str) and f"'id': {input_collection_id}" in x) or
                    (isinstance(x, dict) and x.get('id') == input_collection_id)
                )
            )][['movieId', 'title', 'vote_score', 'genres_list']]
            collection_films = collection_films[
                (collection_films['movieId'].isin(valid_movie_ids)) &
                (~collection_films['movieId'].isin(recommended['movieId'])) &
                (collection_films['genres_list'].apply(lambda x: len(set(x) & set(input_genres)) >= 2))
            ]

        # Si no hay suficientes películas de la colección, usar similitud de texto
        if len(collection_films) < needed:
            extra_needed = needed - len(collection_films)
            cosine_sim = linear_kernel(tfidf_matrix[input_idx:input_idx+1], tfidf_matrix).flatten()
            sim_scores = sorted(list(enumerate(cosine_sim)), key=lambda x: x[1], reverse=True)
            sim_scores = [s for s in sim_scores if s[0] != input_idx and metadata.iloc[s[0]]['movieId'] not in recommended['movieId'].values and metadata.iloc[s[0]]['movieId'] in valid_movie_ids][:extra_needed]
            extra_indices = [i for i, _ in sim_scores]
            extra_films = metadata.iloc[extra_indices][['movieId', 'title', 'vote_score']]
            popular_films = pd.concat([collection_films, extra_films]).sort_values(by='vote_score', ascending=False).head(needed)
        else:
            popular_films = collection_films.sort_values(by='vote_score', ascending=False).head(needed)

        extra_recs = pd.DataFrame({
            'movieId': popular_films['movieId'],
            'title': popular_films['title'],
            'similarity': (popular_films['vote_score'] / popular_films['vote_score'].max() * 80)
        })
        recommended = pd.concat([recommended, extra_recs])

    recommended = recommended.sort_values(by='similarity', ascending=False).head(n_recommendations)
    result = list(zip(recommended['title'], recommended['similarity']))
    result = add_diversity(result, metadata, max_same_collection=2, max_similar_genres=3)[:n_recommendations]
    
    end_time = time.time()
    print(f"Tiempo de ejecución de Collaborative Filtering: {end_time - start_time:.2f} segundos")
    return result

# --------------------------------------
# 5. PRUEBA DE FILTROS
# --------------------------------------
if __name__ == "__main__":
    metadata, ratings, tfidf_matrix = load_and_preprocess_data()
    if metadata is None or ratings is None or tfidf_matrix is None:
        print("Error: No se pudieron cargar los datos. El programa se detendrá.")
        exit()

    test_user_id = 1
    test_movie_title = "The Lord of the Rings: The Fellowship of the Ring"
    n_recommendations = 10

    print(f"\n¿'{test_movie_title}' en metadata? {test_movie_title in metadata['title'].values}")
    print(f"¿User ID {test_user_id} en ratings? {test_user_id in ratings['userId'].values}")

    print("\n🎬 Recomendaciones basadas en contenido:")
    try:
        content_recs = content_based_recommender(test_movie_title, metadata, n_recommendations, tfidf_matrix)
        for i, (title, sim) in enumerate(content_recs, 1):
            print(f"{i}. {title} ({sim:.2f}% de similitud)")
    except ValueError as e:
        print(e)

    print("\n👤 Recomendaciones colaborativas:")
    try:
        collab_recs = collaborative_recommender(test_user_id, ratings, metadata, n_recommendations, test_movie_title, tfidf_matrix)
        for i, (title, sim) in enumerate(collab_recs, 1):
            print(f"{i}. {title} ({sim:.2f}% de relevancia)")
    except ValueError as e:
        print(e)