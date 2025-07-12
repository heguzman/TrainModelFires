from PIL import Image
import numpy as np
from scipy.stats import mode
import sys
import time


def filtro_moda(path_imagen, kernel_size=3):
    try:
        print(f"Inicio proceso del filtro de la moda")
        start_time = time.time()




        # Abrir la imagen con PIL
        imagen = Image.open(path_imagen)
        if imagen.mode not in ('L', 'RGB'):
            raise ValueError("La imagen debe estar en escala de grises (L) o color (RGB)")

        # Convertir a numpy array
        img_array = np.array(imagen)

        # Verificar kernel impar
        if kernel_size % 2 == 0:
            raise ValueError("El tamaño del kernel debe ser impar (3, 5, 7...)")

        # Crear array para la imagen filtrada
        img_filtrada = np.zeros_like(img_array)
        margen = kernel_size // 2
        altura, ancho = img_array.shape[:2]
        canales = 1 if len(img_array.shape) == 2 else img_array.shape[2]

        # Aplicar filtro de la moda
        for i in range(margen, altura - margen):
            for j in range(margen, ancho - margen):
                if canales == 1:  # Escala de grises
                    ventana = img_array[i - margen:i + margen + 1, j - margen:j + margen + 1]
                    img_filtrada[i, j] = mode(ventana.flatten(), keepdims=True)[0][0]
                else:  # Color (RGB)
                    for c in range(canales):
                        ventana = img_array[i - margen:i + margen + 1, j - margen:j + margen + 1, c]
                        img_filtrada[i, j, c] = mode(ventana.flatten(), keepdims=True)[0][0]

        # Convertir de vuelta a imagen PIL y guardar
        imagen_filtrada = Image.fromarray(img_filtrada)
        nombre_salida = path_imagen.replace('.png', '_filtro_moda.png')
        imagen_filtrada.save(nombre_salida)
        print(f"Imagen filtrada guardada como: {nombre_salida}")
        filterModa_time = time.time() - start_time
        print(f"Tiempo del proceso del filtro de la moda: {filterModa_time:.4f} segundos")
        # Mostrar resultados
        imagen.show(title="Original (PIL)")
        imagen_filtrada.show(title="Filtro de la Moda (PIL)")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)