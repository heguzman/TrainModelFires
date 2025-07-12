import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle

def train_classification_models(csv_file_path, target_column, test_size=0.2, random_state=111):

    # Load the dataset
    print(f"Loading dataset from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)

    # Display basic information about the dataset
    print("\nDataset Information:")
    print(f"Shape: {df.shape}")
    print("\nFeature Preview:")
    print(df.head())

    # Extract features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Handle categorical features if any
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"\nEncoding {len(categorical_cols)} categorical features...")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    # Feature scaling (important for Gaussian NB)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    print(f"\nSplit dataset into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples")

    # Initialize different Naive Bayes classifiers
    models = {
        #'Gaussian NB': GaussianNB(),
        'SVML': SVC(kernel='linear', probability=True, random_state=random_state),
        'SVMR': SVC(kernel='rbf', probability=True, random_state=random_state),  # Gaussian Radial Basis Function
        #'SVM_Polynomial': SVC(kernel='poly', probability=True, degree=3, random_state=random_state),
        #'SVM_Sigmoid': SVC(kernel='sigmoid', probability=True, random_state=random_state),
        'LR': LogisticRegression(max_iter=1000, random_state=random_state),
        #'RF': RandomForestClassifier(n_estimators=100, random_state=random_state),
        #'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        #'ET': ExtraTreesClassifier(n_estimators=100, random_state=random_state),
        #'Gradient Boosting Balanced': GradientBoostingClassifier(
        #    n_estimators=150,               # Más árboles que la configuración por defecto
        #    learning_rate=0.1,              # Tasa de aprendizaje estándar
        #    max_depth=5,                    # Profundidad moderada
        #    min_samples_split=10,           # Mínimo de muestras para dividir un nodo
        #    min_samples_leaf=5,             # Mínimo de muestras en nodos hoja
        #    subsample=0.8,                  # Usar el 80% de las muestras para entrenar cada árbol (previene overfitting)
        #    max_features=0.8,               # Usar el 80% de las características para cada árbol
        #    random_state=random_state
        #),
        #'GBP': GradientBoostingClassifier(
        #    n_estimators=300,  # Muchos más árboles para mayor precisión
        #    learning_rate=0.05,  # Tasa de aprendizaje más baja para convergencia más precisa
        #    max_depth=6,  # Un poco más profundo para capturar relaciones complejas
        #    min_samples_split=5,  # Más flexible en la división
        #    min_samples_leaf=3,  # Más flexible en los nodos hoja
        #    subsample=0.9,  # Usar 90% de las muestras
        #    max_features=0.9,  # Usar 90% de las características
        #    validation_fraction=0.1,  # Fracción para validación
        #    n_iter_no_change=15,  # Iteraciones sin mejora antes de detener
        #    tol=1e-4,  # Tolerancia para detener
        #    random_state=random_state
        #),
        'GBPO': GradientBoostingClassifier(
            n_estimators=200,  # Cantidad moderada de árboles
            learning_rate=0.075,  # Tasa de aprendizaje equilibrada
            max_depth=4,  # Profundidad más limitada para prevenir sobreajuste
            min_samples_split=15,  # Requerimiento más estricto para división
            min_samples_leaf=8,  # Más muestras mínimas en hojas
            subsample=0.7,  # Menor muestra para mayor diversidad
            max_features=0.7,  # Menor uso de características para mayor diversidad
            validation_fraction=0.15,  # Mayor fracción para validación
            n_iter_no_change=10,  # Detención temprana
            random_state=random_state
        ),
        'GBF': GradientBoostingClassifier(
            n_estimators=100,  # Menos árboles para mayor velocidad
            learning_rate=0.15,  # Tasa más alta para convergencia rápida
            max_depth=3,  # Árboles poco profundos (más rápidos)
            min_samples_split=20,  # Divisiones más estrictas
            min_samples_leaf=10,  # Hojas con más muestras
            subsample=0.6,  # Menos muestras para mayor velocidad
            max_features=0.6,  # Menos características para mayor velocidad
            random_state=random_state
        ),
        ##'Hist Gradient Boosting': HistGradientBoostingClassifier(max_iter=100, random_state=random_state),
        #'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=random_state),
         #Redes neuronales
        #'Neural Network Relu 2 capas 200 100': MLPClassifier(hidden_layer_sizes=(200, 100), activation='relu', max_iter=1000,
        #                                     solver='adam', random_state=random_state),
        #'Neural Network Relu 2 capas 300 100': MLPClassifier(hidden_layer_sizes=(300, 100), activation='relu', max_iter=1000,
        #                                       solver='adam', random_state=random_state),
        #'NN-Relu-2L(300 200)': MLPClassifier(hidden_layer_sizes=(300, 200), activation='relu',
        #                                                     max_iter=1000,
        #                                                     solver='adam', random_state=random_state),
        #'Neural Network Relu 2 capas 200 100 rate 0.01': MLPClassifier(hidden_layer_sizes=(200, 100), activation='relu',
        #                                                     max_iter=1000,
        #                                                     solver='adam', random_state=random_state, learning_rate_init=0.01),
        #'Neural Network Relu 2 capas 200 100 rate 0.0001': MLPClassifier(hidden_layer_sizes=(200, 100), activation='relu',
        #                                                               max_iter=1000,
        #                                                               solver='adam', random_state=random_state,
        #                                                               learning_rate_init=0.0001),
        #'NN-ReLU-2L(200,100)-r1': MLPClassifier(hidden_layer_sizes=(300, 100), activation='relu',
        #                                                     max_iter=1000,
        #                                                     solver='adam', random_state=random_state,
        #                                                     learning_rate_init=0.01),
        #'Neural Network Relu 2 capas 300 100 rate 0.0001': MLPClassifier(hidden_layer_sizes=(300, 100), activation='relu',
        #                                                               max_iter=1000,
        #                                                               solver='adam', random_state=random_state,
        #                                                               learning_rate_init=0.0001),
        #'Neural Network logistic 2 capas 200 100': MLPClassifier(hidden_layer_sizes=(200, 100), activation='logistic',
        #                                                     max_iter=1000,
        #                                                     solver='adam', random_state=random_state),
        'NN-Log(300,100)-r2': MLPClassifier(hidden_layer_sizes=(300, 100), activation='logistic',
                                                             max_iter=1000,
                                                             solver='adam', random_state=random_state),
        #'Neural Network logistic 2 capas 300 200': MLPClassifier(hidden_layer_sizes=(300, 200), activation='logistic',
        #                                                     max_iter=1000,
        #                                                     solver='adam', random_state=random_state),
        #'Neural Network Relu 3 capas': MLPClassifier(hidden_layer_sizes=(300, 150, 50), activation='relu',
        #                                             max_iter=1000,
        #                                             solver='adam', random_state=random_state),
        #'Neural Network Tanh 2 capas': MLPClassifier(hidden_layer_sizes=(300, 150), activation='tanh',
        #                                             max_iter=1000,
        #                                             solver='adam', random_state=random_state),
        #'Neural Network Tanh 3 capas': MLPClassifier(hidden_layer_sizes=(300, 150, 50), activation='tanh', max_iter=1000,
        #                                     solver='adam', random_state=random_state),
        #'Neural Network Relu 2 capas 200 100': MLPClassifier(hidden_layer_sizes=(200, 100), activation='relu',
        #                                                     max_iter=1000,
        #                                                     solver='adam', random_state=random_state),
        #'Neural Network Relu 2 capas 200 100 rate 0.0001': MLPClassifier(hidden_layer_sizes=(200, 100), activation='relu',
        #                                                              max_iter=1000,
        #                                                              solver='adam', random_state=random_state,
        #                                                              learning_rate_init=0.0001),
        #'Neural Network Relu 2 capas 200 100 rate 0.0001 alpha 0.01': MLPClassifier(hidden_layer_sizes=(200, 100),
        #                                                                 activation='relu',
        #                                                                 max_iter=1000,
        #                                                                 solver='adam', random_state=random_state,
        #                                                                 learning_rate_init=0.0001, alpha=0.01),
        #'Neural Network Relu 2 capas 200 100 rate 0.0001 alpha 0.001': MLPClassifier(hidden_layer_sizes=(200, 100),
        #                                                                            activation='relu',
        #                                                                            max_iter=1000,
        #                                                                            solver='adam',
        #                                                                            random_state=random_state,
        #                                                                            learning_rate_init=0.0001,
        #                                                                            alpha=0.010),
        #'Neural Network Relu 2 capas 200 100 rate 0.0001 early stoping': MLPClassifier(hidden_layer_sizes=(200, 100),
        #                                                                             activation='relu',
        #                                                                             max_iter=1000,
        #                                                                             solver='adam',
        #                                                                             random_state=random_state,
        #                                                                             learning_rate_init=0.0001,
        #                                                                             early_stopping=True,
        #                                                                             validation_fraction=0.2),
        #'Neural Network Relu 2 capas 200 100 rate 0.0001 adaptive 32': MLPClassifier(hidden_layer_sizes=(200, 100),
        #                                                                               activation='relu',
        #                                                                               max_iter=1000,
        #                                                                               solver='adam',
        #                                                                               random_state=random_state,
        #                                                                               learning_rate_init=0.0001,
        #                                                                               learning_rate='adaptive',
        #                                                                               batch_size=32),
        #'Neural Network Relu 2 capas 200 100 rate 0.0001 adaptive 64': MLPClassifier(hidden_layer_sizes=(200, 100),
        #                                                                             activation='relu',
        #                                                                             max_iter=1000,
        #                                                                             solver='adam',
        #                                                                             random_state=random_state,
        #                                                                             learning_rate_init=0.0001,
        #                                                                             learning_rate='adaptive',
        #                                                                             batch_size=64),
        #'Neural Network Relu 2 capas 200 100 rate 0.0001 adaptive 128': MLPClassifier(hidden_layer_sizes=(200, 100),
        #                                                                             activation='relu',
        #                                                                             max_iter=1000,
        #                                                                             solver='adam',
        #                                                                             random_state=random_state,
        #                                                                             learning_rate_init=0.0001,
        #                                                                             learning_rate='adaptive',
        #                                                                             batch_size=128),
        #'Neural Network Relu 3 capas 200 100 50 rate 0.0001': MLPClassifier(hidden_layer_sizes=(200, 100),
        #                                                                              activation='relu',
        #                                                                              max_iter=1000,
        #                                                                              solver='adam',
        #                                                                              random_state=random_state,
        #                                                                              learning_rate_init=0.0001),


    }

    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        try:
            print(f"\nTraining {name}...")

            start_time = time.time()

            model.fit(X_train, y_train)

            training_time = time.time() - start_time
            print(f"Tiempo de entrenamiento para {name}: {training_time:.4f} segundos")

            # Make predictions
            y_pred = model.predict(X_test)

            # Get probability predictions for ROC curve (binary classification)
            if len(np.unique(y)) == 2:
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                except:
                    # Some models might not support predict_proba
                    y_prob = None
            else:
                y_prob = None

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            if len(np.unique(y)) == 2:  # Binary classification
                f1 = f1_score(y_test, y_pred)

                # Calculate AUC if probabilities are available
                auc_score = None
                if y_prob is not None:
                    try:
                        auc_score = roc_auc_score(y_test, y_prob)
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                    except:
                        auc_score = None
                        fpr, tpr = None, None
            else:  # Multi-class
                f1 = f1_score(y_test, y_pred, average='weighted')
                auc_score = None
                fpr, tpr = None, None

            print(f"{name} Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, " +
                  (f"AUC: {auc_score:.4f}" if auc_score is not None else "AUC: N/A"))

            # Generate classification report
            class_report = classification_report(y_test, y_pred)
            print(f"\n{name} Classification Report:")
            print(class_report)

            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'auc': auc_score,
                'classification_report': class_report,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'fpr': fpr,
                'tpr': tpr
            }

            # Save feature importance for Random Forest
            if name == 'Random Forest':
                feature_names = X.columns if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                print("\nTop 10 Important Features:")
                print(feature_importance.head(10))
                results[name]['feature_importance'] = feature_importance

        except Exception as e:
            print(f"Error training {name}: {str(e)}")

    # Identify the best model based on accuracy
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest model: {best_model[0]} with accuracy {best_model[1]['accuracy']:.4f}")

    return results, scaler, X_test, y_test


def plot_confusion_matrices(results, folder, save_path="confusion_matrices.png"):
    """Plot confusion matrices for all models"""
    save_path = os.path.join(folder, save_path)
    # Determine number of models
    n_models = len(results)

    # Calculate number of rows needed (5 models per row)
    n_rows = (n_models + 4) // 3  # Integer ceiling division
    n_cols = min(3, n_models)  # Maximum 5 columns

    # Create figure with appropriate size
    # Height is proportional to number of rows, width is fixed for 5 columns
    plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))

    # Plot each confusion matrix
    for i, (name, result) in enumerate(results.items()):
        # Calculate subplot position
        row = i // 5
        col = i % 5

        # Create subplot
        ax = plt.subplot(n_rows, n_cols, i + 1)

        # Get confusion matrix
        cm = result['confusion_matrix']

        # Convert to percentages (row-wise)
        cm_percentage = np.zeros_like(cm, dtype=float)
        for row_idx in range(cm.shape[0]):
            row_sum = cm[row_idx].sum()
            if row_sum > 0:  # Avoid division by zero
                cm_percentage[row_idx] = (cm[row_idx] / row_sum) * 100

        # Create annotations with percentage values
        annot = []
        for row in cm_percentage:
            annot_row = []
            for val in row:
                # Format as percentage without decimal places
                annot_row.append(f"{int(np.round(val))}%")
            annot.append(annot_row)
        # Convert to numpy array of strings
        annot = np.array(annot, dtype=str)

        # Create heatmap
        sns.heatmap(
            cm,
            annot=annot,  # Show numbers in cells
            fmt='',  # Display as integers
            cmap='Blues',  # Blue color scheme
            cbar=False,  # No color bar to save space
            ax=ax,
            annot_kws={"size": 14}
        )

        # Set title and labels
        ax.set_title(f'{name}', fontsize=16)
        ax.set_xlabel('Predicted', fontsize=14)
        ax.set_ylabel('True', fontsize=14)

        # Try to get class names if possible
        # For binary classification, we can use 0/1 or No Burn/Burn
        if cm.shape[0] == 2:
            ax.set_xticklabels(['Unburned', 'Burned'], fontsize=14)
            ax.set_yticklabels(['Unburned', 'Burned'], fontsize=14)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrices saved to {save_path}")
    plt.close()


def plot_model_comparison(results, folder, save_path="model_comparison.png"):
    """Plot multiple metrics (Accuracy, F1-score, AUC) for all models"""
    save_path = os.path.join(folder, save_path)
    model_names = list(results.keys())
    accuracies = [result['accuracy'] for result in results.values()]
    f1_scores = [result['f1_score'] for result in results.values()]

    # Some models might not have AUC (multi-class or not probability-based)
    aucs = []
    for result in results.values():
        if 'auc' in result and result['auc'] is not None:
            aucs.append(result['auc'])
        else:
            aucs.append(0)  # Use 0 for models without AUC

    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'F1-Score': f1_scores,
        'AUC': aucs
    })

    # Melt the DataFrame for seaborn
    df_melted = pd.melt(df, id_vars=['Model'], value_vars=['Accuracy', 'F1-Score', 'AUC'],
                        var_name='Metric', value_name='Score')

    # Create plot
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted)

    # Add labels
    plt.xlabel('Model', fontsize=14)  # Aumentar tamaño de la etiqueta del eje X
    plt.ylabel('Score Value', fontsize=14)  # Aumentar tamaño de la etiqueta del eje Y
    plt.title('Model Performance Comparison', fontsize=16)  # Aumentar tamaño del título

    # Aumentar tamaño de los nombres de los modelos en el eje X
    plt.xticks(rotation=0, fontsize=12)  # Aumentar el tamaño de la fuente a 12

    # Aumentar tamaño de los valores del eje Y
    plt.yticks(fontsize=12)

    plt.legend(
        title='Metric',
        bbox_to_anchor=(0.5, 0.1),  # (x, y): 0.5=centro horizontal, -0.25=debajo del gráfico
        loc='upper center',  # Punto de anclaje de la leyenda
        ncol=3,  # Tres columnas para distribuir horizontalmente
        fontsize=12,
        title_fontsize=13,  # Aumentar tamaño del título de la leyenda
        frameon=True
    )
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=14)  # Tamaño de las etiquetas de las barras

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison saved to {save_path}")
    plt.close()


def plot_roc_curves(results, folder, save_path="roc_curves.png"):
    """Plot ROC curves for models that support probability prediction"""
    save_path = os.path.join(folder, save_path)
    plt.figure(figsize=(10, 8))

    has_valid_curves = False  # Flag to check if we have at least one valid curve

    for name, result in results.items():
        if ('fpr' in result and result['fpr'] is not None and
                'tpr' in result and result['tpr'] is not None and
                'auc' in result and result['auc'] is not None):
            plt.plot(result['fpr'], result['tpr'],
                     label=f'{name} (AUC = {result["auc"]:.2f})')
            has_valid_curves = True

    # Only add the rest of the plot elements if we have at least one valid curve
    if has_valid_curves:
        # Add the random classifier line
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Different Models')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)

        plt.savefig(save_path)
        print(f"ROC curves saved to {save_path}")
    else:
        print("No valid ROC curves to plot - skipping ROC curve generation")

    plt.close()

def plot_feature_importance(feature_importance_df, folder, save_path="feature_importance.png", top_n=10):
    """
    Plot top N important features from Random Forest

    Parameters:
    feature_importance_df: DataFrame with feature importance values
    save_path: Path to save the plot image file
    top_n: Number of top features to display
    """
    save_path = os.path.join(folder, save_path)
    top_features = feature_importance_df.head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {top_n} Feature Importance (Random Forest)')
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Feature importance plot saved to {save_path}")
    plt.close()


def predict_with_best_model(model, new_data, scaler=None):
    """Make predictions with the best model"""
    if scaler is not None:
        new_data = scaler.transform(new_data)
    return model.predict(new_data)





def trainClassificationModel(csv_file_path, folder):
    target_column = "Burn_Classification"

    # Train the models
    results, scaler, X_test, y_test = train_classification_models(csv_file_path, target_column)

    # Plot confusion matrices
    if results:
        plot_confusion_matrices(results, folder)
        plot_model_comparison(results, folder)

        # Plot ROC curves for binary classification
        if len(np.unique(y_test)) == 2:
            plot_roc_curves(results, folder)

        # Plot feature importance if Random Forest was trained
        if 'Random Forest' in results and 'feature_importance' in results['Random Forest']:
            plot_feature_importance(
                results['Random Forest']['feature_importance'],
               folder, "feature_importance.png"
            )

        # Extract the best model
        best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_model = results[best_model_name]['model']

        ## Save the best model
        #model_path = os.path.join(folder, f"{best_model_name}_model.pkl")
        #with open(model_path, 'wb') as f:
        #    pickle.dump(best_model, f)
#
        #print(f"\nBest model ({best_model_name}) saved to {model_path}")
        # Save the scaler
        scaler_path = os.path.join(folder, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")
        # Save All models separeted
        for name, result in results.items():
            model_path = os.path.join(folder, f"{name}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)



        print(f"\nYou can now use the best model ({best_model_name}) for predictions!")

        return results