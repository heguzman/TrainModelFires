from PIL import Image
import numpy as np
from scipy import stats
import time


def filtro_moda_manhattan(path_imagen, radio=1):
    print(f"Inicio proceso del filtro de la moda")
    start_time = time.time()

    # Abrir la imagen con PIL
    imagen = Image.open(path_imagen)

    if imagen.mode not in ('L', 'RGB'):
        raise ValueError("La imagen debe estar en escala de grises (L) o color (RGB)")

    # Convertir a numpy array
    img_array = np.array(imagen)

    altura, ancho = img_array.shape[:2]
    canales = 1 if len(img_array.shape) == 2 else img_array.shape[2]

    # Crear una copia para la imagen filtrada
    if canales == 1:
        img_filtrada = np.zeros((altura, ancho), dtype=img_array.dtype)
    else:
        img_filtrada = np.zeros((altura, ancho, canales), dtype=img_array.dtype)

    # Aplicar el filtro
    for i in range(radio, altura - radio):
        for j in range(radio, ancho - radio):
            if canales == 1:  # Escala de grises
                valores = []
                for di in range(-radio, radio + 1):
                    for dj in range(-radio, radio + 1):
                        # Condición de distancia Manhattan: |di| + |dj| <= radio
                        if abs(di) + abs(dj) <= radio:
                            valores.append(img_array[i + di, j + dj])
                img_filtrada[i, j] = stats.mode(valores, keepdims=True)[0][0]
            else:  # Color (RGB)
                for c in range(canales):
                    valores = []
                    for di in range(-radio, radio + 1):
                        for dj in range(-radio, radio + 1):
                            # Condición de distancia Manhattan: |di| + |dj| <= radio
                            if abs(di) + abs(dj) <= radio:
                                valores.append(img_array[i + di, j + dj, c])
                    img_filtrada[i, j, c] = stats.mode(valores, keepdims=True)[0][0]

    imagen_filtrada = Image.fromarray(img_filtrada)
    nombre_salida = path_imagen.replace('.png', '_filtro_moda_manhattan.png')
    imagen_filtrada.save(nombre_salida)
    print(f"Imagen filtrada guardada como: {nombre_salida}")
    filterModa_time = time.time() - start_time
    print(f"Tiempo del proceso del filtro de la moda: {filterModa_time:.4f} segundos")
    # Mostrar resultados
    imagen.show(title="Original (PIL)")
    imagen_filtrada.show(title="Filtro de la Moda (PIL)")