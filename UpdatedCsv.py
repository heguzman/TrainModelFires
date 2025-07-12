import csv
import os


def filter_csv_columns(input_file, output_file, columns_to_keep):
    """
    Filtra un archivo CSV para mantener solo las columnas especificadas.

    Args:
        input_file (str): Ruta al archivo CSV de entrada
        output_file (str): Ruta donde guardar el archivo CSV filtrado
        columns_to_keep (list): Lista de nombres de columnas a mantener
    """
    try:
        # Crear el directorio si no existe
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Se ha creado el directorio: '{output_dir}'")

        with open(input_file, 'r', newline='') as infile:
            reader = csv.DictReader(infile)

            # Verificar que todas las columnas solicitadas existen en el archivo
            all_columns = reader.fieldnames
            for column in columns_to_keep:
                if column not in all_columns:
                    print(f"Error: La columna '{column}' no existe en el archivo original.")
                    print(f"Columnas disponibles: {all_columns}")
                    return False

            with open(output_file, 'w', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=columns_to_keep)
                writer.writeheader()

                for row in reader:
                    # Crear un nuevo diccionario solo con las columnas requeridas
                    filtered_row = {col: row[col] for col in columns_to_keep}
                    writer.writerow(filtered_row)

        print(f"Archivo procesado correctamente. Resultado guardado en '{output_file}'")
        return True

    except FileNotFoundError:
        print(f"Error: El archivo '{input_file}' no se encuentra.")
        return False
    except Exception as e:
        print(f"Error inesperado: {str(e)}")
        return False


def process_csv(columns_to_keep, output_file, filerOriginal="data/raw/totalComplete.csv"):
    filter_csv_columns(filerOriginal, output_file, columns_to_keep)