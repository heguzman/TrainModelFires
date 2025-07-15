from kfp.v2.dsl import component, Input, Output, Dataset

@component(
    base_image="python:3.10"
)
def filter_csv_columns_component(
    input_file: str,
    output_file: str,
    columns_to_keep: list
):
    import csv
    import os

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
                filtered_row = {col: row[col] for col in columns_to_keep}
                writer.writerow(filtered_row)

    print(f"Archivo procesado correctamente. Resultado guardado en '{output_file}'")
    return True