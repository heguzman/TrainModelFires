from itertools import combinations

def combined_Columns(listColumnsComplete):
    # Column fix
    fixed_column = 'Burn_Classification'

    optional_columns = [col for col in listColumnsComplete if col != fixed_column]

    all_combinations = []

    # Generate all combinations of the optional columns
    for r in range(1, len(optional_columns) + 1):
        # Obtener combinaciones de r elementos de las columnas opcionales
        for combo in combinations(optional_columns, r):
            # Añadir la columna fija a cada combinación
            current_combination = list(combo) + [fixed_column]
            all_combinations.append(current_combination)

    return all_combinations