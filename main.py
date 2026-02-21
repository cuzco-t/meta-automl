import os
import openml
import numpy as np
import pandas as pd

import logging
import warnings

from scipy import sparse
from pathlib import Path
from typing import Iterator

from src import Result, ExtractorMetaFeatures, MineroDePipelines, BaseDeDatos
from src.PipelineLogger import PipelineLogger


def get_archivos(carpeta: str) -> list[Path]:
    """
    Obtiene las rutas completas de los archivos .txt en la carpeta especificada.
    
    :return: Una lista de objetos Path con las rutas completas de los archivos .txt encontrados en la carpeta.
    :rtype: list[Path]
    """
    carpeta_path = Path(carpeta)
    
    rutas_archivos = []
    # Iterar sobre todos los archivos en la carpeta
    for archivo in os.listdir(carpeta_path):
        ruta_completa = (carpeta_path / archivo).resolve()
        
        if ruta_completa.suffix != ".txt":
            continue

        rutas_archivos.append(ruta_completa)

    return rutas_archivos

def get_generador_tasks_ids(ruta_archivo: Path) -> Iterator[int]:
    """
    Lee un archivo de texto línea por línea y devuelve un generador 
    que produce cada número entero encontrado en el archivo.
    
    :param ruta_archivo: La ruta del archivo de texto que contiene los números enteros, uno por línea.
    :type ruta_archivo: Path
    :return: Un generador que produce cada número entero encontrado en el archivo.
    :rtype: Iterator[int]
    """
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            
            # Si la línea está vacía, se omite
            if not linea:
                continue
            
            yield int(linea)

def get_datos_openml(
    task_id: int
) -> Result[
    tuple[str, pd.DataFrame, pd.Series],
    str
]:
    """
    Obtiene los datos de OpenML para una tarea dada por su ID.
    
    :param task_id: El ID de la tarea en OpenML para la cual se desean obtener los datos.
    :type task_id: int
    :return: Un objeto Result que contiene una tupla con el nombre del dataset,
     el DataFrame de características (X) y la Serie de etiquetas (y) si la operación
     fue exitosa, o un mensaje de error si la operación falló.
    :rtype: Result
    """

    def asegurar_dataframe(X) -> pd.DataFrame | None:
        if isinstance(X, pd.DataFrame):
            return X
        if sparse.issparse(X):
            return pd.DataFrame.sparse.from_spmatrix(X)
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        
        return None

    def asegurar_series(y) -> pd.Series | None:
        if y is None or isinstance(y, pd.Series):
            return y
        if isinstance(y, np.ndarray):
            return pd.Series(y)
        
        return None

    openml_task = None
    try:
        openml_task = openml.tasks.get_task(task_id)
        dataset = openml_task.get_dataset()
    
    except openml.exceptions.OpenMLServerException:
        return Result.fail("Problemas con el servidor de OpenML")
    
    except openml.exceptions.OpenMLPrivateDatasetError:
        return Result.fail("El dataset solicitado es privado")
    
    except Exception:
        return Result.fail("Error desconocido al obtener datos de OpenML")
    
    dataset_name = "Sin nombre"
    if hasattr(dataset, "name"):
        dataset_name = dataset.name

    target_name = None
    if hasattr(dataset, "default_target_attribute"):
        target_name = dataset.default_target_attribute

    X, y, _, _ = dataset.get_data(target=target_name)

    X = asegurar_dataframe(X)
    y = asegurar_series(y)

    if X is None or y is None:
        return Result.fail("No se pudo convertir X o y a DataFrame/Series")
    
    return Result.ok((dataset_name, X, y))

def guardar_dataset(
        X: pd.DataFrame, 
        y: pd.Series, 
        tarea: str, 
        nombre_dataset: str, 
        ruta_base: str
    ) -> None:
    """
    Guarda el dataset en formato CSV en la ruta especificada, organizando por tarea.
    """
    df = pd.concat([X, y], axis=1)
    
    # Crear carpeta de la tarea dentro de la base si no existe
    ruta_tarea = os.path.join(ruta_base, tarea)
    os.makedirs(ruta_tarea, exist_ok=True)
    
    # Ruta completa del archivo
    ruta_archivo = os.path.join(ruta_tarea, f"{nombre_dataset}.csv")
    
    # Guardar CSV
    df.to_csv(ruta_archivo, index=False)
    
    return None

def vectorizar_pipeline(pipeline: dict) -> list[float]:
    fases_acciones_indice = {
        "tratar_duplicados": {
            "eliminar": 184,
            None: 185
        },
        "tratar_faltantes_numericos": {
            "aleatorio": 186,
            "eliminar": 187,
            "media": 188,
            "media_geometrica": 189,
            "mediana": 190,
            "moda": 191,
            None: 192
        },
        "tratar_faltantes_strings": {
            "aleatorio": 193,
            "eliminar": 194,
            "etiqueta_desconocido": 195,
            "moda": 196,
        },
        "codificar_variables_binarias": {
            "label_encoding": 197,
            None: 198
        },
        "codificar_variables_categoricas_rango_bajo": {
            "label_encoding": 199,
            "one_hot_encoding": 200,
        },
        "codificar_variables_categoricas_rango_medio": {
            "eliminar_variable": 201,
            "frequency_encoding": 202,
        },
        "codificar_variables_categoricas_rango_alto": {
            "eliminar_columna": 203
        },
        "tratar_outliers_numericos": {
            "aleatorio": 204,
            "eliminar": 205,
            "media": 206,
            "media_geometrica": 207,
            "mediana": 208,
            "moda": 209,
            None: 210
        },
        "escalar_datos_numericos": {
            "max_abs_scaler": 211,
            "min_max": 212,
            None: 213,
            "robust_scaler": 214,
            "standard_scaler": 215,
        },
        "normalizar_datos_numericos": {
            "box_cox": 216,
            "cuadrado": 217,
            "inverso": 218,
            "ln": 219,
            None: 220,
            "sqrt": 221,
            "z_score": 222,
        },
        "crear_nueva_variable": {
            "llm": 223,
            None: 224
        },
        "seleccionar_variables": {
            "select_from_model": 225,
            "llm": 226,
            "mutual_info": 227,
            None: 228,
            "pca_90": 229,
            "pca_95": 230,
            "pca_99": 231,
            "umap_20": 232,
            "umap_50": 233,
            "umap_80": 234,
            "variance_threshold": 235,
        }
    }

    for fase, acciones in fases_acciones_indice.items():
        for accion, indice in acciones.items():
            # Ajustar índices para que comiencen desde 0
            fases_acciones_indice[fase][accion] = indice - 184

    cantidad_total_acciones = sum(len(acciones) for acciones in fases_acciones_indice.values())
    acciones_vectorizadas = [0.0] * cantidad_total_acciones
    for fase, accion in pipeline.items():
        indice = fases_acciones_indice[fase][accion]
        acciones_vectorizadas[indice] = 1.0

    return acciones_vectorizadas

def promediar_metricas(metricas: dict[str, dict[str, list]]) -> dict[str, float]:
    if metricas is None:
        return {}

    promedios = {
        ejecucion: (sum(valores) / len(valores) if len(valores) == 3 else 0.0)
        for numero_ejecucion in metricas.values()
        for ejecucion, valores in numero_ejecucion.items()
    }

    return promedios


def main():
    RUTA_CARPETA_IDENTIFICADORES = "./data/datasets_identificadores/"
    RUTA_CARPETA_DATSETS_DESCARGADOS = "./data/"

    # Configuración única
    logger = PipelineLogger().get_logger()

    extractor = ExtractorMetaFeatures()
    minero = MineroDePipelines()
    db = BaseDeDatos()

    contador = 0
    for ruta_archivo in get_archivos(RUTA_CARPETA_IDENTIFICADORES):
        tarea_pipeline = ruta_archivo.stem.split("_")[0]

        print("=" * 100)
        print(f"Tarea: {tarea_pipeline}")
        print("=" * 100)

        for task_id in get_generador_tasks_ids(ruta_archivo):
            result_datos_openml = get_datos_openml(task_id)

            if not result_datos_openml.is_success:
                print(f"Error al obtener datos para Task ID {task_id}: {result_datos_openml.get_error()}")

                continue

            dataset_name, X, y = result_datos_openml.get_value()
            guardar_dataset(X, y, tarea_pipeline, dataset_name, RUTA_CARPETA_DATSETS_DESCARGADOS)

            # print("Tipos de datos obtenidos:")
            # print("X dtype:", X.dtypes)
            # print("y dtype:", y.dtype)

            # print("=" * 100)
            # print("Meta-features progreso:")
            # meta_features, meta_features_vectorizadas = extractor.extraer_desde_dataframe(
            #     X, 
            #     y, 
            #     vectorizar=False
            # )

            print("=" * 100)
            print(f"Dataset: {dataset_name}")
            print(f"Task ID: {task_id}")
            print("X shape:", X.shape)
            print("y shape:", y.shape)
            # print("Meta-features extraídas:", meta_features)
            print("=" * 100)

            if task_id == 10:
                print("Task ID:", task_id)

            pipeline, metricas, tiempo_total = minero.pipeline_supervisado(X, y, tarea_pipeline)

            pipeline_vectorizado = vectorizar_pipeline(pipeline)
            # metricas_promediadas = promediar_metricas(metricas)

            #TODO: Formatear segun sea exito o error, y guardar en base de datos
            print("")
            if metricas is None:
                print("Pipeline mal configurado")
            
            print(f"Dataset: {dataset_name}")
            print("Pipeline supervisado finalizado.")
            # print(f"Tiempo de ejecución: {tiempo_total:.2f} segundos")
            # print("Promedios de métricas:")
            # for metrica, valores in metricas.items():
            #     promedio = sum(valores) / len(valores)
            #     print(f"\t{metrica:<20}: {promedio}")

            contador += 1
            # if contador >= 10:  # Limitar a los primeros 5 datasets para la demo
            #     return
            
            # input("Presiona Enter para continuar con el siguiente dataset...")

if __name__ == "__main__":
    main()