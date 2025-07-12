import pandas as pd
from PIL import Image
import numpy as np

def printImage(csv_predicted, csv_origin, modelName, folder="test_1"):
    """
    Genera una imagen a partir de un archivo CSV que contiene coordenadas y valores de píxeles.

    Args:
        csv_predicted (str): Ruta al archivo CSV con las predicciones.
        csv_origin (str): Ruta al archivo CSV original para X e Y.
    """
    # Cargar los dos archivos CSV
    csvOrigin = pd.read_csv(csv_origin)  # CSV con columnas X e Y
    csvPRedicted = pd.read_csv(csv_predicted)  # CSV con columna predicted

    # Verificar que ambos tienen la misma cantidad de filas
    if len(csvOrigin) != len(csvPRedicted):
        print("¡Advertencia! Los archivos CSV tienen diferente número de filas")
        print(f"CSV1: {len(csvOrigin)} filas, CSV2: {len(csvPRedicted)} filas")

    # Extraer las columnas necesarias
    df_final = pd.DataFrame()
    df_final['X'] = csvOrigin['X']
    df_final['Y'] = csvOrigin['Y']
    df_final['prediction'] = csvPRedicted['prediction']


    # Encontrar las dimensiones máximas
    ancho = int(df_final['X'].max() + 1)
    alto = int(df_final['Y'].max() + 1)

    # Crear una imagen vacía
    imagen = Image.new('RGB', (ancho, alto), color='white')
    pixels = imagen.load()

    # Asignar colores a cada píxel
    for _, fila in df_final.iterrows():
        x = int(fila['X'])
        y = int(fila['Y'])
        valor = int(fila['prediction'])

        if valor == 0:
            color = (0, 255, 0)  # Verde en RGB
        elif valor == 1:
            color = (255, 0, 0)  # Rojo en RGB
        elif valor == 2:
            color = (0, 0, 255)  # Azul en RGB
        else:
            color = (255, 255, 255)  # Blanco para valores desconocidos

        pixels[x, y] = color

    # Guardar la imagen
    imagen.save(f'{folder}/imagen_generada{modelName}.png')

    # Mostrar la imagen (opcional)
    # imagen.show()
