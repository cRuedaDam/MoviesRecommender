# test_imports.py
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
print("Python path:", sys.path)
try:
    from src.data.loader import load_data
    from src.data.preprocessor import preprocess_data
    from src.recommenders.content_based import content_based_recommender
    from src.recommenders.collaborative import collaborative_recommender
    print("Importaciones exitosas")
except Exception as e:
    print(f"Error en las importaciones: {e}")
try:
    import src.data
    print(f"Módulo src.data encontrado en: {src.data.__path__}")
except ImportError:
    print("No se pudo importar el módulo src.data")