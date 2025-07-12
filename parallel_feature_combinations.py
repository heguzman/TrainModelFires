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
    process_csv(columnsImage, f"{folder}/dataImage.csv", "ImageComplete.csv")
    csv_image = f"{folder}/dataImage.csv"
    scaler = f"{folder}/scaler.pkl"
    model = f"{folder}/GBPO_model.pkl"
    modelName = "GBPO"
    predict_with_trained_model(model, scaler, csv_image, modelName, folder)
    printImage(f"{folder}/image_predicted_{modelName}.csv", "ImageComplete.csv", modelName, folder)
    return metrics

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

def run_parallel_combinations(start_index=1):
    columns = ['NORMALIZED_NDBI', 'NORMALIZED_VV', 'NORMALIZED_VH', 'RBR_NDBI', 'RBR_VV','RBR_VH', 'Post_VV', 'Post_VH', 'Pre_VV', 'Pre_VH', 'BAI', 'BAIM', 'EVI2', 'NBR', 'NDVI', 'Burn_Classification']
    columnsCombinedList = combined_Columns(columns)
    all_metrics = []
    numTest = 1
    columnsCombinedList.append(columns)
    combinations_args = [(column, i+1) for i, column in enumerate(columnsCombinedList) if i+1 >= start_index]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_combination, combinations_args))
    for metrics in results:
        all_metrics.extend(metrics)
    combine_all_metrics() 