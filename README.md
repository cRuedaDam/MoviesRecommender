# 🎬 Movie Recommender System

Bienvenido al Movie Recommender System, un proyecto que te ayuda a descubrir películas basadas en tus gustos. Este sistema combina dos enfoques: recomendaciones basadas en contenido y filtrado colaborativo, todo presentado a través de una interfaz interactiva con Streamlit y posters obtenidos directamente de la API de TMDB. 🚀

## 📖 ¿Qué hace este proyecto?

Este sistema recomienda películas utilizando dos métodos distintos:

### Recomendaciones basadas en contenido 🎥

**¿Cómo funciona?** Encuentra películas similares a una que elijas, basándose en características como género, saga, año, idioma, popularidad y puntuación.

**Pesos de las características:**
- Similitud (TF-IDF de descripciones): 30% 📝
- Colección (sagas): 15% 📚
- Género: 10% 🎭
- Género principal: 15% 🌟
- Año de estreno: 5% 📅
- Idioma: 5% 🌍
- Popularidad: 10% 🔥
- Puntuación: 5% ⭐

**Técnica:** Usa `linear_kernel` de scikit-learn para calcular la similitud coseno en una matriz TF-IDF:
```python
cosine_similarities = linear_kernel(tfidf_matrix[movie_idx:movie_idx+1], tfidf_matrix).flatten()
```

**Resultado:** Devuelve las 10 películas más similares con un porcentaje de similitud y sus posters oficiales.

### Filtrado colaborativo 🤝

**¿Cómo funciona?** Recomienda películas según las valoraciones de usuarios similares, prediciendo qué te podría gustar aunque no hayas visto esas películas.

**Técnica:** Utiliza el algoritmo K-Nearest Neighbors (KNN) de scikit-learn para encontrar usuarios con gustos similares.

**Resultado:** Sugiere películas con altas puntuaciones predichas basadas en las valoraciones de tus "vecinos".

### Integración con TMDB API 🎨

**¿Cómo funciona?** El sistema se conecta a la API de The Movie Database (TMDB) para obtener posters oficiales de alta calidad de las películas recomendadas.

**Beneficios:**
- Posters oficiales y actualizados
- Interfaz visual más atractiva
- Información visual que facilita la identificación de películas

### Interfaz con Streamlit 🌐

Todo se visualiza en una aplicación web local creada con Streamlit. Selecciona una película para recomendaciones basadas en contenido o introduce tus preferencias para filtrado colaborativo, con posters visuales para cada recomendación.

## 📂 Estructura del proyecto

```
MoviesRecommender/
├── app.py                # Script principal de la app Streamlit
├── dataset/              # Datasets utilizados
│   ├── movies_metadata.csv   # Metadatos de películas (título, géneros, etc.)
│   ├── links.csv         # Mapeo de IDs entre plataformas
│   └── ratings_small.csv # Valoraciones de usuarios
├── src/                  # Código fuente
│   ├── data/             # Módulos para carga y preprocesamiento
│   │   ├── __init__.py
│   │   ├── loader.py     # Carga los datasets
│   │   └── preprocessor.py   # Preprocesa los datos
│   └── recommenders/     # Lógica de recomendación
│       ├── __init__.py
│       ├── content_based.py  # Recomendador basado en contenido
│       ├── collaborative.py  # Recomendador colaborativo
│       └── utils.py      # Funciones de utilidad y API de TMDB
```

Nota: Los directorios `__pycache__` están excluidos en `.gitignore` y no están en el repositorio.

## 🛠 Requisitos

Para ejecutar el proyecto, necesitas:

- Python 3.8 o superior
- pip para instalar dependencias
- Un entorno virtual (opcional, pero recomendado)
- **Clave API de TMDB** (gratuita en https://www.themoviedb.org/settings/api)

Dependencias clave:
- pandas
- numpy
- scikit-learn
- streamlit
- scipy
- requests (para conexión con API de TMDB)

## 🚀 Instalación

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/cRuedaDam/MoviesRecommender.git
   cd MoviesRecommender
   ```

2. **Crea un entorno virtual (opcional):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instala las dependencias:**

   Ejecuta:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configura la API de TMDB:**
   - Regístrate en https://www.themoviedb.org/
   - Obtén tu clave API gratuita en https://www.themoviedb.org/settings/api
   - Configura la clave en tu aplicación (variable de entorno o archivo de configuración)

5. **Verifica los datasets:**
   Asegúrate de que los archivos `movies_metadata.csv`, `links.csv` y `ratings_small.csv` estén en la carpeta `dataset/`.

## 🎮 Cómo ejecutar la aplicación

1. **Configura tu clave API de TMDB:**
   Asegúrate de tener configurada tu clave API de TMDB antes de ejecutar la aplicación.

2. **Inicia la app de Streamlit:**
   Desde la raíz del proyecto, ejecuta:
   ```bash
   streamlit run app.py
   ```

3. **Accede a la interfaz:**
   - Streamlit abrirá un servidor local (normalmente en http://localhost:8501).
   - Abre la URL en tu navegador para interactuar con el sistema.

4. **Usa la app:**
   - **Recomendaciones basadas en contenido:** Selecciona una película y obtén las 10 más similares con sus posters oficiales.
   - **Filtrado colaborativo:** Introduce tu ID de usuario o valoraciones para recomendaciones personalizadas con visualización de posters.

## 💡 Consejos de uso

- Asegúrate de que los datasets estén en la carpeta `dataset/` para que el sistema funcione.
- **Configura correctamente tu clave API de TMDB** para mostrar los posters de las películas.
- El filtrado colaborativo con KNN puede requerir ajustes (e.g., número de vecinos) para mejores resultados.
- Usa un equipo con suficiente memoria, ya que la matriz TF-IDF y el entrenamiento de KNN pueden ser intensivos.
- La conexión a internet es necesaria para obtener los posters de TMDB.

## 🔮 Mejoras futuras

- Mejoras en la interfaz de Streamlit (filtros por año, género, etc.).
- Despliegue en la nube para acceso remoto.
- Integración de más información de TMDB (trailers, reparto, etc.).
- Sistema de fallback para posters no disponibles.

## ⚙️ Configuración de TMDB API

Para obtener tu clave API de TMDB:

1. Crea una cuenta en https://www.themoviedb.org/
2. Ve a tu perfil → Configuración → API
3. Solicita una clave API (es gratuita)
4. Configura la clave en tu aplicación

La API de TMDB proporciona:
- Posters de alta calidad
- Información actualizada de películas
- Múltiples tamaños de imagen
- Acceso gratuito con límites razonables

## 📬 Contacto

¿Tienes preguntas o ideas? ¡Abre un issue en el repositorio o contáctame en [cruedadam@gmail.com]!

⭐ ¡Si te gusta el proyecto, déjame una estrella en GitHub! ⭐