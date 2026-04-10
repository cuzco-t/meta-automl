import numpy as np
import pandas as pd

from typing import Any, Dict, List

from ..Result import Result
from scipy.optimize import linear_sum_assignment

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, median_absolute_error, explained_variance_score, r2_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

from datetime import datetime

print_original = print
def print(*args, **kwargs):
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_original(f"{ahora} |", *args, **kwargs)

class EvaluadorModelos:
    """Clase para evaluar modelos de machine learning según la tarea."""
    
    def _calcular_metricas_clasificacion(self, y_val: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas para tareas de clasificación."""
        print("Iniciando evaluacion de clasificación...")
        y_val_str = y_val.astype(str)
        y_pred_str = y_pred.astype(str)

        acc = accuracy_score(y_val_str, y_pred_str)
        prec = precision_score(y_val_str, y_pred_str, average="weighted", zero_division=0)
        rec = recall_score(y_val_str, y_pred_str, average="weighted", zero_division=0)
        f1 = f1_score(y_val_str, y_pred_str, average="weighted", zero_division=0)

        print("OK - Evaluacion de clasificacion completada.")

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
    
    def _calcular_metricas_regresion(self, y_val: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas para tareas de regresión de forma robusta."""

        print("Iniciando evaluacion de regresion...")
        try:
            # Validaciones básicas
            if y_val is None or y_pred is None:
                raise ValueError("Inputs None")

            if len(y_val) == 0 or len(y_pred) == 0:
                raise ValueError("Inputs vacíos")

            if len(y_val) != len(y_pred):
                raise ValueError("Tamaños inconsistentes")

            # Convertir a numpy
            y_val = np.asarray(y_val)
            y_pred = np.asarray(y_pred)

            # Métricas
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)

            mae = mean_absolute_error(y_val, y_pred)
            medae = median_absolute_error(y_val, y_pred)
            ev = explained_variance_score(y_val, y_pred)

            # R2 puede ser problemático
            try:
                r2 = r2_score(y_val, y_pred)
                if np.isnan(r2):
                    r2 = -1.0
            except Exception:
                r2 = -1.0

            print("OK - Evaluacion de regresion completada.")
            return {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "medae": medae,
                "ev": ev
            }

        except Exception:
            # Valores por defecto seguros
            print("ERROR - Evaluacion de regresion fallida")
            return {
                "mae": 999.0,
                "mse": 999.0,
                "rmse": 999.0,
                "r2": -1.0,
                "medae": 999.0,
                "ev": -1.0
            }

    def _calcular_metricas_clustering(self, X_val: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas para tareas de clustering."""
        
        print("Iniciando evaluacion de clustering...")

        # Validación inicial
        invalid_labels = (
            y_pred is None or
            len(y_pred) == 0 or
            len(set(y_pred)) < 2
        )

        # --- Silhouette ---
        try:
            if invalid_labels:
                silhouette_score_val = -1.0
            else:
                val = silhouette_score(X_val, y_pred)
                if val is None or np.isnan(val):
                    silhouette_score_val = -1.0
                else:
                    silhouette_score_val = val
        except Exception:
            silhouette_score_val = -1.0

        # --- Calinski-Harabasz ---
        try:
            if invalid_labels:
                calinski_score_val = 0.0
            else:
                val = calinski_harabasz_score(X_val, y_pred)
                if val is None or np.isnan(val):
                    calinski_score_val = 0.0
                else:
                    calinski_score_val = val
        except Exception:
            calinski_score_val = 0.0

        # --- Davies-Bouldin ---
        try:
            if invalid_labels:
                davies_score_val = 999.0
            else:
                val = davies_bouldin_score(X_val, y_pred)
                if val is None or np.isnan(val):
                    davies_score_val = 999.0
                else:
                    davies_score_val = val
        except Exception:
            davies_score_val = 999.0

        print("OK - Evaluacion de clustering completada.")
        return {
            "silhouette": silhouette_score_val,
            "calinski": calinski_score_val,
            "davies": davies_score_val
        }
    
    def _obtener_metricas_fallo(self, tarea: str) -> Dict[str, float]:
        """Retorna métricas con valores de fallo según la tarea."""
        if tarea.lower() == "clasificacion":
            return {"accuracy": -1.0, "precision": -1.0, "recall": -1.0, "f1": -1.0}
        elif tarea.lower() == "regresion":
            return {"mae": 999.0, "mse": 999.0, "rmse": 999.0, "r2": -1.0, "medae": 999.0, "ev": -1.0}
        elif tarea.lower() == "clustering":
            return {"silhouette": -1.0, "calinski": 0.0, "davies": 999.0}
        return {}
    
    def evaluar_modelos(
        self,
        lista_results_modelos: List[List[Result]],
        folds_data: Dict[int, Dict[str, pd.DataFrame | pd.Series]],
        tarea: str
    ) -> List[Dict[str, float]]:
        """
        Evalúa una lista de modelos usando validación cruzada.
        
        Args:
            lista_results_modelos: Lista de listas de objetos Result con modelos
            folds_data: Diccionario con datos de validación por fold
            tarea: Tipo de tarea ('clasificacion', 'regresion', 'clustering')
            
        Returns:
            Lista de diccionarios con métricas promediadas por modelo
        """
        resultados_evaluacion = []
        
        for resultado_list in lista_results_modelos:
            # Verificar si algún resultado en la lista es fallo
            if any(resultado.is_failure for resultado in resultado_list):
                resultados_evaluacion.append({
                    "estado": "CRASH", 
                    "error": resultado_list[0].get_error()
                })
                continue
            
            metricas_folds = {fold_id: {} for fold_id in folds_data.keys()}
            
            # Evaluar en cada fold
            fallo_en_evaluacion = False
            for fold_id, fold_data in folds_data.items():
                X_val = fold_data["X_val"]
                y_val = fold_data["y_val"]

                # Obtener el modelo correspondiente al fold (índice fold_id - 1)
                modelo = resultado_list[fold_id - 1].get_value()

                # Predicción
                try:
                    y_pred = modelo.predict(X_val)
                
                except Exception as e:
                    fallo_en_evaluacion = True
                    for fold in metricas_folds.keys():
                        metricas_folds[fold] = self._obtener_metricas_fallo(tarea)
                    break

                # Calcular métricas según tarea
                if tarea.lower() == "clasificacion":
                    metricas_folds[fold_id] = self._calcular_metricas_clasificacion(y_val, y_pred)
                elif tarea.lower() == "regresion":
                    metricas_folds[fold_id] = self._calcular_metricas_regresion(y_val, y_pred)
                elif tarea.lower() == "clustering":
                    metricas_folds[fold_id] = self._calcular_metricas_clustering(X_val, y_pred)
            
            # Promediar métricas de los folds
            metricas_promedio = {}
            todas_las_metricas = set().union(*[set(m.keys()) for m in metricas_folds.values()])
            
            for metrica in todas_las_metricas:
                valores = [metricas_folds[fold_id][metrica] for fold_id in metricas_folds.keys()]
                metricas_promedio[metrica] = float(np.mean(valores))
            
            resultados_evaluacion.append(metricas_promedio)
        
        return resultados_evaluacion

    def evaluar_modelos_clustering(
        self,
        lista_modelos_result: List[Result],
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[Dict[str, float]]:
        """
        Evalúa modelos de clustering calculando métricas de clustering y clasificación.
        
        Args:
            lista_modelos_result: Lista de objetos Result con modelos de clustering
            X: DataFrame con las características
            y: Series con las etiquetas verdaderas
            
        Returns:
            Lista de diccionarios con métricas de clustering y clasificación
        """
        resultados_evaluacion = []
        
        for modelo_result in lista_modelos_result:
            # Verificar si el resultado es fallo
            if modelo_result.is_failure:
                resultados_evaluacion.append({
                    "estado": "CRASH", 
                    "error": modelo_result.get_error()
                })
                continue
            
            try:
                # Obtener las etiquetas predichas (usar el primer resultado disponible)
                y_pred = modelo_result.get_value().labels_
                
                # Calcular métricas de clustering
                metricas_clustering = self._calcular_metricas_clustering(X, y_pred)
                
                # Convertir todo a string (opcional pero recomendable)
                y_true = y.astype(str).to_numpy()
                y_pred = np.array(y_pred).astype(str)

                clases_unicas = np.unique(y_true)
                clusters_unicos = np.unique(y_pred)

                # Crear mapeos seguros
                map_clases = {clase: idx for idx, clase in enumerate(clases_unicas)}
                map_clusters = {cluster: idx for idx, cluster in enumerate(clusters_unicos)}

                # Crear matriz de confusión
                matriz_confusion = np.zeros((len(clusters_unicos), len(clases_unicas)))

                for i in range(len(y_true)):
                    fila = map_clusters[y_pred[i]]
                    columna = map_clases[y_true[i]]
                    matriz_confusion[fila, columna] += 1
                
                # Aplicar algoritmo húngaro (maximizar coincidencias)
                row_ind, col_ind = linear_sum_assignment(-matriz_confusion)
                
                # Mapear índices internos
                cluster_to_clase = {
                    clusters_unicos[row]: clases_unicas[col]
                    for row, col in zip(row_ind, col_ind)
                }

                y_pred_mapeado = np.array([
                    cluster_to_clase.get(label, label)
                    for label in y_pred
                ])
                
                # Calcular métricas de clasificación
                metricas_clasificacion = self._calcular_metricas_clasificacion(y, y_pred_mapeado)
                
                # Combinar métricas
                metricas_totales = {**metricas_clustering, **metricas_clasificacion}
                resultados_evaluacion.append(metricas_totales)
                
            except Exception as e:
                resultados_evaluacion.append({
                    "estado": "CRASH", 
                    "error": str(e)
                })

        return resultados_evaluacion
