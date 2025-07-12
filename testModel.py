import pandas as pd
import numpy as np
import pickle
import time
from sklearn.preprocessing import StandardScaler


def predict_with_trained_model(model_path, scaler_path, new_csv_path, modelName, folder, target_column=None):
    """
    Realiza predicciones usando un modelo previamente entrenado en un nuevo conjunto de datos.

    Args:
        model_path (str): Ruta al archivo del modelo guardado (.pkl)
        scaler_path (str): Ruta al archivo del scaler guardado (.pkl)
        new_csv_path (str): Ruta al nuevo CSV para realizar predicciones
        target_column (str, optional): Nombre de la columna objetivo si está presente en el CSV

    Returns:
        pandas.DataFrame: DataFrame con los datos originales y las predicciones
    """
    # Cargar el modelo y el scaler
    print(f"Cargando modelo desde {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"Cargando scaler desde {scaler_path}...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Cargar el nuevo dataset
    print(f"Cargando nuevo dataset desde {new_csv_path}...")
    df_new = pd.read_csv(new_csv_path)

    ## Preparar los datos para la predicción
    #if target_column and target_column in df_new.columns:
    #    X_new = df_new.drop(target_column, axis=1)
    #    y_true = df_new[target_column]
    #    has_target = True
    #else:
    #    X_new = df_new
    #    has_target = False

    ## Manejar características categóricas si hay alguna
    #categorical_cols = X_new.select_dtypes(include=['object']).columns
    #if len(categorical_cols) > 0:
    #    print(f"Codificando {len(categorical_cols)} características categóricas...")
    #    for col in categorical_cols:
    #        # Nota: Aquí deberías usar el mismo LabelEncoder que usaste durante el entrenamiento
    #        # Si no lo guardaste, esto podría causar inconsistencias
    #        from sklearn.preprocessing import LabelEncoder
    #        le = LabelEncoder()
    #        X_new[col] = le.fit_transform(X_new[col])

    # Aplicar la misma escala que en el entrenamiento
    X_scaled = scaler.transform(df_new)

    # Realizar predicciones
    print("Realizando predicciones...")
    start_time = time.time()

    predictions = model.predict(X_scaled)

    prediction_time = time.time() - start_time
    print(f"Tiempo de predicción: {prediction_time:.4f} segundos")

    # Si el modelo admite probabilidades y es clasificación binaria, obtenerlas
    probabilities = None
    try:
        probabilities = model.predict_proba(X_scaled)
    except:
        pass

    # Crear DataFrame con resultados
    result_df = df_new.copy()
    result_df['prediction'] = predictions

    if probabilities is not None and probabilities.shape[1] == 2:
        result_df['probability'] = probabilities[:, 1]

    result_df.to_csv(f"{folder}/image_predicted_{modelName}.csv", index=False)
    print(f"Resultados guardados en 'resultados_prediccion.csv'")



    return result_df