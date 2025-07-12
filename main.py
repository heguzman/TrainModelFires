from UpdatedCsv import process_csv
from columnsCombined import combined_Columns
from filtroModa import filtro_moda
from filtroModaManhattan import filtro_moda_manhattan
from plotImage import printImage
from testModel import predict_with_trained_model
from trainClassificationModel import trainClassificationModel
import pandas as pd
import os
import concurrent.futures
import glob

#columns = ['NDPI', 'NDBI', 'NORMALIZED_NDBI', 'NORMALIZED_VV', 'NORMALIZED_VH', 'RBR_NDBI', 'RBR_VV','RBR_VH', 'Burn_Classification']

#MISMA PRUEBA QUE EL PAPE
#columns = ['HV Post', 'VH Post', 'IRV', 'NDPI', 'Burn_Classification']
columns = ['NORMALIZED_NDBI', 'NORMALIZED_VV', 'NORMALIZED_VH', 'RBR_NDBI', 'RBR_VV','RBR_VH', 'Post_VV', 'Post_VH', 'Pre_VV', 'Pre_VH', 'BAI', 'BAIM', 'EVI2', 'NBR', 'NDVI', 'Burn_Classification']
#columns = ['NORMALIZED_NDBI', 'NORMALIZED_VV', 'NORMALIZED_VH', 'RBR_NDBI', 'RBR_VV','RBR_VH', 'Post_VV', 'Post_VH', 'Pre_VV', 'Pre_VH', 'Burn_Classification']
#columns = ['BAI', 'BAIM', 'EVI2', 'NBR', 'NDVI', 'Burn_Classification']
columnsCombinedList = combined_Columns(columns)
resultTotal = []
all_metrics = []
# Process the CSV file
numTest = 1
#columnsCombinedList = []
# Generate all combinations of columns
#test1 = ['NDPI', 'NDBI', 'NORMALIZED_VH', 'RBR_NDBI', 'RBR_VV', 'Burn_Classification']
#columnsCombinedList.append(test1)
#test2 = ['NDPI', 'NDBI', 'NORMALIZED_VH', 'RBR_VV', 'RBR_VH', 'Burn_Classification']
#columnsCombinedList.append(test2)
#test3 = ['RBR_VV', 'RBR_VH', 'NORMALIZED_VH', 'NORMALIZED_NDBI', 'RBR_NDBI', 'Burn_Classification']
#columnsCombinedList.append(test3)
#test4 = ['NDPI', 'NDBI', 'NORMALIZED_VV', 'NORMALIZED_VH', 'NORMALIZED_NDBI', 'Burn_Classification']
#columnsCombinedList.append(test4)
#test5 = ['NDPI', 'NDBI', 'NORMALIZED_VH', 'NORMALIZED_NDBI', 'RBR_VV', 'RBR_NDBI', 'Burn_Classification']
columnsCombinedList.append(columns)

# Llama a la versión paralela si se desea
from parallel_feature_combinations import run_parallel_combinations

def process_combination(args):
    column, numTest = args
    base_folder = "pruebas"
    folder = os.path.join(base_folder, f"test_{numTest}")
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    output_file = f"{folder}/total.csv"
    process_csv(column, output_file)
    results = trainClassificationModel(output_file, folder)
    model_names = list(results.keys())
    metrics = []
    for model in model_names:
        metric_row = {
            'Num Test': numTest,
            'Model': model,
            'Accuracy': results[model]['accuracy'],
            'F1 Score': results[model]['f1_score'],
            'AUC': results[model].get('auc', 'N/A'),
            'Features': ','.join([f for f in column if f != 'Burn_Classification'])
        }
        metrics.append(metric_row)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(folder, "metrics.csv"), index=False)
    columnsImage = column.copy()
    if 'Burn_Classification' in columnsImage:
        columnsImage.remove('Burn_Classification')
    process_csv(columnsImage, f"{folder}/dataImage.csv", "data/raw/ImageComplete.csv")
    csv_image = f"{folder}/dataImage.csv"
    scaler = f"{folder}/scaler.pkl"
    model = f"{folder}/GBPO_model.pkl"
    modelName = "GBPO"
    predict_with_trained_model(model, scaler, csv_image, modelName, folder)
    printImage(f"{folder}/image_predicted_{modelName}.csv", "data/raw/ImageComplete.csv", modelName, folder)
    return metrics

# Función para combinar todos los CSV de métricas

def combine_all_metrics():
    all_metrics = []
    output_folder = "pruebas"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for metrics_file in glob.glob("pruebas/test_*/metrics.csv"):
        df = pd.read_csv(metrics_file)
        all_metrics.append(df)
    if all_metrics:
        final_df = pd.concat(all_metrics, ignore_index=True)
        final_df.to_csv(os.path.join(output_folder, "model_metrics.csv"), index=False)
        print(f"Metrics saved to {os.path.join(output_folder, 'model_metrics.csv')}")
    else:
        print("No metrics files found.")

if __name__ == "__main__":
    import sys
    start_index = 1
    if len(sys.argv) > 1:
        start_index = int(sys.argv[1])
    run_parallel_combinations(start_index)
    combine_all_metrics()
    # Si quieres usar el código original, comenta la línea de arriba y descomenta el bloque siguiente:
    # columnsCombinedList = combined_Columns(columns)
    # resultTotal = []
    # all_metrics = []
    # numTest = 1
    # columnsCombinedList.append(columns)
    # for column in columnsCombinedList:
    #     ...
    # results = list(executor.map(process_combination, combinations_args))
    # # results es una lista de listas de métricas, las aplanamos
    # for metrics in results:
    #     all_metrics.extend(metrics)
    # # Guardar las métricas en un archivo CSV al finalizar todas las combinaciones
    # metrics_df = pd.DataFrame(all_metrics)
    # metrics_df.to_csv('model_metrics.csv', index=False)
    # print("Metrics saved to model_metrics.csv")
    # exit(0)

## Test training models with complete image
#columnsImage = ['NORMALIZED_NDBI', 'NORMALIZED_VV', 'NORMALIZED_VH', 'RBR_NDBI', 'RBR_VV','RBR_VH', 'Post_VV', 'Post_VH', 'Pre_VV', 'Pre_VH', 'BAI', 'BAIM', 'EVI2', 'NBR', 'NDVI']
##process_csv(columnsImage, "test_1/dataImage.csv", "ImageComplete.csv")
##columnsImage = ['BAI', 'BAIM', 'EVI2', 'NBR', 'NDVI']
#process_csv(columnsImage, "test_1/dataImage.csv", "ImageComplete.csv")
##MODELO ADABOOST
#folder = f"test_{numTest}"
#csv_image = "test_1/dataImage.csv"
#scaler = "test_1/scaler.pkl"

##MODELO Neural Network logistic 2 capas 300 100_model
#model = "test_1/GBF_model.pkl"
#modelName="GBF_model"
#predict_with_trained_model(model, scaler, csv_image, modelName)
#printImage(f"image_predicted_{modelName}.csv", "ImageComplete.csv", modelName)
#filtro_moda(f"imagen_generada{modelName}.png")
##filtro_moda_manhattan(f"imagen_generada{modelName}.png")



#model = "test_1/SVMR_model.pkl"
#modelName="SVMR"
#predict_with_trained_model(model, scaler, csv_image, modelName)
#printImage(f"image_predicted_{modelName}.csv", "ImageComplete.csv", modelName)
#filtro_moda(f"imagen_generada{modelName}.png")
##filtro_moda_manhattan(f"imagen_generada{modelName}.png")

##MODELO Gradient Boosting Fast
#model = "test_1/SVML_model.pkl"
#modelName="SVML"
#predict_with_trained_model(model, scaler, csv_image, modelName)
#printImage(f"image_predicted_{modelName}.csv", "ImageComplete.csv", modelName)
#filtro_moda(f"imagen_generada{modelName}.png")
##filtro_moda_manhattan(f"imagen_generada{modelName}.png")

##MODELO Gradient Boosting Prevent Overfitting
#model = "test_1/GBPO_model.pkl"
#modelName="GBPO"
#predict_with_trained_model(model, scaler, csv_image, modelName)
#printImage(f"image_predicted_{modelName}.csv", "ImageComplete.csv", modelName)
#filtro_moda(f"imagen_generada{modelName}.png")
##filtro_moda_manhattan(f"imagen_generada{modelName}.png")

##MODELO Logistic Regression
#model = "test_1/NN-Log(300,100)-r2_model.pkl"
#modelName="NN-Log(300,100)-r2_model"
#predict_with_trained_model(model, scaler, csv_image, modelName)
#printImage(f"image_predicted_{modelName}.csv", "ImageComplete.csv", modelName)
#filtro_moda(f"imagen_generada{modelName}.png")
##filtro_moda_manhattan(f"imagen_generada{modelName}.png")

##MODELO Neural Network logistic 2 capas 200 100
#model = "test_1/LR_model.pkl"
#modelName="LR_model"
#predict_with_trained_model(model, scaler, csv_image, modelName)
#printImage(f"image_predicted_{modelName}.csv", "ImageComplete.csv", modelName)
#filtro_moda(f"imagen_generada{modelName}.png")
##filtro_moda_manhattan(f"imagen_generada{modelName}.png")



##MODELO Neural Network logistic 2 capas 300 200_model
#model = "test_1/Neural Network logistic 2 capas 300 200_model.pkl"
#modelName="Neural_Network_logistic_2_capas_300_200"
#predict_with_trained_model(model, scaler, csv_image, modelName)
#printImage(f"image_predicted_{modelName}.csv", "ImageComplete.csv", modelName)
#filtro_moda(f"imagen_generada{modelName}.png")
#filtro_moda_manhattan(f"imagen_generada{modelName}.png")