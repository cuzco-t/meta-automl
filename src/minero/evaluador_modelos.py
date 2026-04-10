import numpy as np
import pandas as pd
import multiprocessing as mp

from typing import Any, Dict, List
from queue import Empty

from ..Result import Result
from scipy.optimize import linear_sum_assignment

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, median_absolute_error, explained_variance_score, r2_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

from datetime import datetime

TIMEOUT_EVALUACION_SEGUNDOS = 300

print_original = print
def print(*args, **kwargs):
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_original(f"{ahora} |", *args, **kwargs)


def _metricas_fallo_por_tarea(tarea: str) -> Dict[str, float]:
    tarea_normalizada = tarea.lower()
    if tarea_normalizada == "clasificacion":
        return {"accuracy": -1.0, "precision": -1.0, "recall": -1.0, "f1": -1.0}
    if tarea_normalizada == "regresion":
        return {"mae": 999.0, "mse": 999.0, "rmse": 999.0, "r2": -1.0, "medae": 999.0, "ev": -1.0}
    if tarea_normalizada == "clustering":
        return {"silhouette": -1.0, "calinski": 0.0, "davies": 999.0}
    return {}


def _metricas_fallo_clustering_completo() -> Dict[str, float]:
    return {
        "silhouette": -1.0,
        "calinski": 0.0,
        "davies": 999.0,
        "accuracy": -1.0,
        "precision": -1.0,
        "recall": -1.0,
        "f1": -1.0,
    }


def _calcular_metricas_clasificacion_worker(y_val: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    y_val_str = y_val.astype(str)
    y_pred_str = np.asarray(y_pred).astype(str)

    acc = accuracy_score(y_val_str, y_pred_str)
    prec = precision_score(y_val_str, y_pred_str, average="weighted", zero_division=0)
    rec = recall_score(y_val_str, y_pred_str, average="weighted", zero_division=0)
    f1 = f1_score(y_val_str, y_pred_str, average="weighted", zero_division=0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


def _calcular_metricas_regresion_worker(y_val: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    try:
        if y_val is None or y_pred is None:
            raise ValueError("Inputs None")

        if len(y_val) == 0 or len(y_pred) == 0:
            raise ValueError("Inputs vacios")

        if len(y_val) != len(y_pred):
            raise ValueError("Tamanos inconsistentes")

        y_val_np = np.asarray(y_val)
        y_pred_np = np.asarray(y_pred)

        mse = mean_squared_error(y_val_np, y_pred_np)
        rmse = np.sqrt(mse)

        mae = mean_absolute_error(y_val_np, y_pred_np)
        medae = median_absolute_error(y_val_np, y_pred_np)
        ev = explained_variance_score(y_val_np, y_pred_np)

        try:
            r2 = r2_score(y_val_np, y_pred_np)
            if np.isnan(r2):
                r2 = -1.0
        except Exception:
            r2 = -1.0

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "medae": medae,
            "ev": ev
        }
    except Exception:
        return _metricas_fallo_por_tarea("regresion")


def _calcular_metricas_clustering_worker(X_val: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, float]:
    invalid_labels = (
        y_pred is None or
        len(y_pred) == 0 or
        len(set(y_pred)) < 2
    )

    try:
        if invalid_labels:
            silhouette_score_val = -1.0
        else:
            val = silhouette_score(X_val, y_pred)
            silhouette_score_val = -1.0 if val is None or np.isnan(val) else val
    except Exception:
        silhouette_score_val = -1.0

    try:
        if invalid_labels:
            calinski_score_val = 0.0
        else:
            val = calinski_harabasz_score(X_val, y_pred)
            calinski_score_val = 0.0 if val is None or np.isnan(val) else val
    except Exception:
        calinski_score_val = 0.0

    try:
        if invalid_labels:
            davies_score_val = 999.0
        else:
            val = davies_bouldin_score(X_val, y_pred)
            davies_score_val = 999.0 if val is None or np.isnan(val) else val
    except Exception:
        davies_score_val = 999.0

    return {
        "silhouette": silhouette_score_val,
        "calinski": calinski_score_val,
        "davies": davies_score_val
    }


def _evaluar_fold_worker(
    cola_resultado: mp.Queue,
    fold_id: int,
    modelo: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    tarea: str
) -> None:
    try:
        y_pred = modelo.predict(X_val)

        tarea_normalizada = tarea.lower()
        if tarea_normalizada == "clasificacion":
            metricas = _calcular_metricas_clasificacion_worker(y_val, y_pred)
        elif tarea_normalizada == "regresion":
            metricas = _calcular_metricas_regresion_worker(y_val, y_pred)
        else:
            metricas = _metricas_fallo_por_tarea(tarea)

        cola_resultado.put({"ok": True, "fold_id": fold_id, "metricas": metricas})
    except Exception as e:
        cola_resultado.put({"ok": False, "fold_id": fold_id, "error": str(e)})


def _evaluar_modelo_clustering_worker(
    cola_resultado: mp.Queue,
    modelo: Any,
    X: pd.DataFrame,
    y: pd.Series
) -> None:
    try:
        y_pred = modelo.labels_

        metricas_clustering = _calcular_metricas_clustering_worker(X, y_pred)

        y_true = y.astype(str).to_numpy()
        y_pred_str = np.array(y_pred).astype(str)

        clases_unicas = np.unique(y_true)
        clusters_unicos = np.unique(y_pred_str)

        map_clases = {clase: idx for idx, clase in enumerate(clases_unicas)}
        map_clusters = {cluster: idx for idx, cluster in enumerate(clusters_unicos)}

        matriz_confusion = np.zeros((len(clusters_unicos), len(clases_unicas)))
        for i in range(len(y_true)):
            fila = map_clusters[y_pred_str[i]]
            columna = map_clases[y_true[i]]
            matriz_confusion[fila, columna] += 1

        row_ind, col_ind = linear_sum_assignment(-matriz_confusion)
        cluster_to_clase = {
            clusters_unicos[row]: clases_unicas[col]
            for row, col in zip(row_ind, col_ind)
        }

        y_pred_mapeado = np.array([
            cluster_to_clase.get(label, label)
            for label in y_pred_str
        ])

        metricas_clasificacion = _calcular_metricas_clasificacion_worker(y, y_pred_mapeado)
        metricas_totales = {**metricas_clustering, **metricas_clasificacion}

        cola_resultado.put({"ok": True, "metricas": metricas_totales})
    except Exception as e:
        cola_resultado.put({"ok": False, "error": str(e)})

class EvaluadorModelos:
    """Clase para evaluar modelos de machine learning según la tarea."""
        
    def _obtener_metricas_fallo(self, tarea: str) -> Dict[str, float]:
        """Retorna métricas con valores de fallo según la tarea."""
        return _metricas_fallo_por_tarea(tarea)
    
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

            # Evaluar en cada fold en procesos separados
            procesos = {}
            colas = {}

            for fold_id, fold_data in folds_data.items():
                X_val = fold_data["X_val"]
                y_val = fold_data["y_val"]

                # Obtener el modelo correspondiente al fold (índice fold_id - 1)
                modelo = resultado_list[fold_id - 1].get_value()

                cola_resultado = mp.Queue()
                proceso = mp.Process(
                    target=_evaluar_fold_worker,
                    args=(cola_resultado, fold_id, modelo, X_val, y_val, tarea)
                )
                proceso.start()

                procesos[fold_id] = proceso
                colas[fold_id] = cola_resultado

            for fold_id in folds_data.keys():
                proceso = procesos[fold_id]
                cola_resultado = colas[fold_id]

                proceso.join(TIMEOUT_EVALUACION_SEGUNDOS)

                if proceso.is_alive():
                    proceso.terminate()
                    proceso.join()
                    print(f"TIMEOUT en fold {fold_id}. Se asignan metricas por defecto.")
                    metricas_folds[fold_id] = self._obtener_metricas_fallo(tarea)
                    cola_resultado.close()
                    cola_resultado.join_thread()
                    continue

                try:
                    resultado = cola_resultado.get_nowait()
                except Empty:
                    resultado = {"ok": False, "error": "Sin resultado en cola"}

                if resultado.get("ok"):
                    metricas_folds[fold_id] = resultado["metricas"]
                else:
                    print(f"ERROR en fold {fold_id}: {resultado.get('error', 'Error desconocido')}")
                    metricas_folds[fold_id] = self._obtener_metricas_fallo(tarea)

                cola_resultado.close()
                cola_resultado.join_thread()
            
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
                modelo = modelo_result.get_value()

                cola_resultado = mp.Queue()
                proceso = mp.Process(
                    target=_evaluar_modelo_clustering_worker,
                    args=(cola_resultado, modelo, X, y)
                )
                proceso.start()

                proceso.join(TIMEOUT_EVALUACION_SEGUNDOS)

                if proceso.is_alive():
                    proceso.terminate()
                    proceso.join()
                    print("TIMEOUT en evaluacion de clustering. Se asignan metricas por defecto.")
                    resultados_evaluacion.append(_metricas_fallo_clustering_completo())
                    cola_resultado.close()
                    cola_resultado.join_thread()
                    continue

                try:
                    resultado = cola_resultado.get_nowait()
                except Empty:
                    resultado = {"ok": False, "error": "Sin resultado en cola"}

                if resultado.get("ok"):
                    resultados_evaluacion.append(resultado["metricas"])
                else:
                    print(f"ERROR en evaluacion de clustering: {resultado.get('error', 'Error desconocido')}")
                    resultados_evaluacion.append(_metricas_fallo_clustering_completo())

                cola_resultado.close()
                cola_resultado.join_thread()
                
            except Exception as e:
                print(f"ERROR preparando evaluacion de clustering: {str(e)}")
                resultados_evaluacion.append(_metricas_fallo_clustering_completo())

        return resultados_evaluacion
