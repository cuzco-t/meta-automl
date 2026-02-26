import numpy as np
import pandas as pd

from typing import Any, Dict, List

from ..Result import Result

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, median_absolute_error, explained_variance_score, r2_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)


class EvaluadorModelos:
    """Clase para evaluar modelos de machine learning según la tarea."""
    
    def _calcular_metricas_clasificacion(self, y_val: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas para tareas de clasificación."""
        return {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0)
        }
    
    def _calcular_metricas_regresion(self, y_val: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas para tareas de regresión."""
        
        mse = mean_squared_error(y_val, y_pred)
        return {
            "mae": mean_absolute_error(y_val, y_pred),
            "mse": mse,
            "rmse": np.sqrt(mse),
            "r2": r2_score(y_val, y_pred),
            "medae": median_absolute_error(y_val, y_pred),
            "ev": explained_variance_score(y_val, y_pred)
        }
    
    def _calcular_metricas_clustering(self, X_val: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas para tareas de clustering."""
        
        return {
            "silhouette": silhouette_score(X_val, y_pred),
            "calinski": calinski_harabasz_score(X_val, y_pred),
            "davies": davies_bouldin_score(X_val, y_pred)
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
                resultados_evaluacion.append(self._obtener_metricas_fallo(tarea))
                continue
            
            metricas_folds = {fold_id: {} for fold_id in folds_data.keys()}
            
            # Evaluar en cada fold
            for fold_id, fold_data in folds_data.items():
                X_val = fold_data["X_val"]
                y_val = fold_data["y_val"]
                
                X_train = fold_data["X_train"]

                # # Homogeneizar tipos y orden
                # X_val = X_val.astype(X_train.dtypes.to_dict())  # asegurar mismo dtype
                # X_val = X_val[X_train.columns]                 # asegurar mismo orden

                # Obtener el modelo correspondiente al fold (índice fold_id - 1)
                modelo = resultado_list[fold_id - 1].get_value()

                # Predicción
                try:
                    y_pred = modelo.predict(X_val)
                
                except ValueError as e:
                    if "The feature names should match those that were passed during fit." in str(e):
                        for fold in metricas_folds.keys():
                            metricas_folds[fold] = self._obtener_metricas_fallo(tarea)
                        break
                    if "Input X contains NaN" in str(e):
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