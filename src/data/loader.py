import pandas as pd

def load_data(data_folder='dataset'):
    """
    Carga los archivos CSV de películas, ratings y enlaces.
    
    Args:
        data_folder (str): Carpeta donde están los datos.
        
    Returns:
        tuple: (metadata, ratings, links) o (None, None, None) si hay error.
    """
    try:
        metadata = pd.read_csv(f'{data_folder}/movies_metadata.csv', low_memory=False)
        ratings = pd.read_csv(f'{data_folder}/ratings_small.csv')
        links = pd.read_csv(f'{data_folder}/links.csv')
        return metadata, ratings, links
        
    except FileNotFoundError as e:
        print(f"Error al cargar los archivos: {e}")
        return None, None, None