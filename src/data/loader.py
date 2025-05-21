import pandas as pd

def load_data(data_folder='dataset'):
    """
    Carga los archivos CSV con los datos necesarios para el sistema de recomendación.
    
    Argumentos:
        data_folder (str): Ruta de la carpeta donde se encuentran los archivos CSV.
        
    Retorna:
        tuple: Tres DataFrames con la metadata de películas, valoraciones y enlaces.
               En caso de error, retorna (None, None, None).
    """
    try:
        # Carga la metadata de las películas, desactivando advertencias por mezcla de tipos
        metadata = pd.read_csv(f'{data_folder}/movies_metadata.csv', low_memory=False)
        
        # Carga los ratings de usuarios con las valoraciones
        ratings = pd.read_csv(f'{data_folder}/ratings_small.csv')
        
        # Carga los enlaces (pueden ser IDs para combinar con otras bases de datos)
        links = pd.read_csv(f'{data_folder}/links.csv')
        
        return metadata, ratings, links
        
    except FileNotFoundError as e:
        # Indica qué archivo faltó o no se pudo abrir
        print(f"Error al cargar los archivos: {e}")
        return None, None, None
