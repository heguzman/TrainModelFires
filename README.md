# TrainModelFires

Este proyecto está orientado al entrenamiento y evaluación de modelos de machine learning para identificar zonas incendiadas a partir de datos satelitales SAR y Opticos. Incluye scripts para el procesamiento de datos, entrenamiento, evaluación, y visualización de resultados.

---

## Descripción general

El flujo principal del proyecto consiste en:
1. Procesar y filtrar los datos de entrada.
2. Generar combinaciones de características para el entrenamiento.
3. Entrenar y evaluar distintos modelos de clasificación.
4. Realizar predicciones y visualizarlas espacialmente.

---

## Estructura del proyecto

- `main.py`: Ejecuta el flujo completo de procesamiento, entrenamiento y evaluación en paralelo.
- `trainClassificationModel.py`: Entrena y evalúa modelos de clasificación sobre un conjunto de datos.
- `testModel.py`: Realiza predicciones usando modelos previamente entrenados.
- `columnsCombined.py`: Genera combinaciones de columnas/características para experimentación.
- `filtroModa.py` y `filtroModaManhattan.py`: Aplican filtros estadísticos a los datos.
- `parallel_feature_combinations.py`: Permite probar combinaciones de características en paralelo.
- `plotImage.py`: Visualiza los resultados de las predicciones en formato de imagen.
- `UpdatedCsv.py`: Filtra y transforma archivos CSV según las columnas requeridas.
- `data/`: Carpeta donde se almacenan los archivos de datos de entrada.

---

## Archivos requeridos en `data/`

El proyecto espera encontrar al menos los siguientes archivos en la carpeta `data/`:

- **`data/totalComplete.csv`**  
  Archivo principal de entrada. Debe contener los datos tabulares con las variables necesarias para el entrenamiento y evaluación de los modelos.  
  Columnas esperadas (pueden variar según la combinación):
  - `NORMALIZED_NDBI`, `NORMALIZED_VV`, `NORMALIZED_VH`, `RBR_NDBI`, `RBR_VV`, `RBR_VH`, `Post_VV`, `Post_VH`, `Pre_VV`, `Pre_VH`, `BAI`, `BAIM`, `EVI2`, `NBR`, `NDVI`, `Burn_Classification`
  - **`Burn_Classification`** es la columna objetivo (indica si la zona está incendiada o no).

- **`data/ImageComplete.csv`**  
  Archivo utilizado para la generación de imágenes a partir de las predicciones.  
  Debe contener al menos las columnas `X` y `Y` (coordenadas espaciales) y cualquier otra columna relevante para la visualización.

---

## Instalación

1. Clona este repositorio:
   ```bash
   git clone <URL-del-repositorio>
   cd TrainModelFires
   ```
2. (Opcional) Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Linux/Mac:
   source venv/bin/activate
   ```
3. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```
   > Si no existe el archivo `requirements.txt`, puedes generarlo con las siguientes dependencias sugeridas:
   >
   > pandas, numpy, scikit-learn, matplotlib, seaborn, pillow

---

## Uso

### Ejecución del flujo principal

```bash
python main.py
```

### Entrenamiento de un modelo específico

```bash
python trainClassificationModel.py
```

### Evaluación y predicción

```bash
python testModel.py
```

> Ajusta los comandos según los argumentos o configuraciones que requiera cada script.

---

## Flujo recomendado

1. Preprocesar los datos con los scripts de filtrado y combinación.
2. Entrenar el modelo con `trainClassificationModel.py` o mediante `main.py`.
3. Evaluar el modelo y realizar predicciones con `testModel.py`.
4. Visualizar los resultados con `plotImage.py`.

---

## Ejemplo de dependencias (`requirements.txt`)

```
pandas
numpy
scikit-learn
matplotlib
seaborn
pillow
```

---

## Créditos y licencia

Autor: [Genérico]

Licencia: [Indicar aquí la licencia, por ejemplo MIT, GPL, etc.]