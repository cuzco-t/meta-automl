import os
import json
import openml
import numpy as np
import pandas as pd

import logging
import warnings

from scipy import sparse
from pathlib import Path
from typing import Iterator

from src import Result, ExtractorMetaFeatures, BaseDeDatos
from src.minero.MineroDePipelines import MineroDePipelines
from src.PipelineLogger import PipelineLogger
from src.minero.ejecutor_preprocesamiento import EjecutorPreprocesamiento


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
    tuple[str, str, pd.DataFrame, pd.Series],
    str
]:
    """
    Obtiene los datos de OpenML para una tarea dada por su ID.
    
    :param task_id: El ID de la tarea en OpenML para la cual se desean obtener los datos.
    :type task_id: int
    :return: Un objeto Result que contiene una tupla con el nombre del dataset,
     la descripción del dataset, el DataFrame de características (X) y la Serie 
     de etiquetas (y) si la operación fue exitosa, o un mensaje de error si la operación falló.
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

    descripcion = ""
    if hasattr(dataset, "description"):
        descripcion = dataset.description

    X, y, _, _ = dataset.get_data(target=target_name)

    X = asegurar_dataframe(X)
    y = asegurar_series(y)

    if X is None or y is None:
        return Result.fail("No se pudo convertir X o y a DataFrame/Series")
    
    return Result.ok((dataset_name, descripcion, X, y))

def crear_mapa_indices() -> tuple[dict, int]:
    """
    Crear un mapa de índices para cada fase del pipeline, cada algoritmo dentro de cada fase,
    y cada modelo ML dentro de cada fase.
    
    :return: Un diccionario con los mapas de índices y el número total de dimensiones.
    :rtype: tuple[dict, int]
    """

    contador = 0
    mapa_indices = {}

    tareas = ["clasificacion", "regresion", "clustering"]

    ejecutor = EjecutorPreprocesamiento(PipelineLogger().get_logger())
    fases_instancias = ejecutor.crear_fases_instnacias()

    for fase, instancia in fases_instancias.items():
        algoritmos = instancia.ALGORITMOS
        mapa_indices[fase] = {}
        for algoritmo in algoritmos:
            mapa_indices[fase][algoritmo] = contador
            contador += 1

    minero = MineroDePipelines()

    selectores_modelos = minero.tarea_modelos
    for tarea in tareas:
        for algoritmo in selectores_modelos[tarea]:
            mapa_indices.setdefault(f"modelos_{tarea}", {})[algoritmo] = contador
            contador += 1

    return mapa_indices, contador - 1

def vectorizar_pipeline(mapa_indices, dimensiones, tarea, pipeline, modelo) -> list:
    """
    Crea una lista de vectores paso a paso usando copias del último estado.
    
    Cada elemento de la lista representa el estado del vector después de cada activación.
    """
    vector = [0.0] * dimensiones
    historia = []

    # Activamos índices de cada fase/algoritmo paso a paso
    for fase, algoritmo in pipeline.items():
        vector[mapa_indices[fase][algoritmo]] = 1.0
        historia.append(vector.copy())

    # Activamos el índice del modelo
    indice_modelo = mapa_indices[f"modelos_{tarea}"][modelo]
    vector[indice_modelo] = 1.0
    historia.append(vector.copy())

    return historia

def main():
    RUTA_CARPETA_IDENTIFICADORES = "./data/datasets_identificadores/"
    RUTA_CARPETA_DATSETS_DESCARGADOS = "./data/"

    # Configuración única
    logger = PipelineLogger().get_logger()

    mapa_indices, dimensiones = crear_mapa_indices()

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
                logger.error(
                    "Descargar de dataset fallida", 
                    extra={
                        "task_id": task_id, 
                        "tarea": tarea_pipeline, 
                        "error": result_datos_openml.get_error()
                    }
                )
                continue

            dataset_name, descripcion, X, y = result_datos_openml.get_value()

            logger.info(
                "Dataset descargado exitosamente",
                extra={
                    "task_id": task_id,
                    "tarea": tarea_pipeline,
                    "dataset_name": dataset_name
                }
            )
            
            logger.info(
                "Iniciando extraccion de meta-features",
                extra={
                    "task_id": task_id,
                    "tarea": tarea_pipeline,
                    "dataset_name": dataset_name
                }
            )
            meta_features, meta_features_vectorizadas = extractor.extraer_desde_dataframe(
                X, 
                y, 
                vectorizar=True
            )

            logger.info(
                "Meta-features extraidas exitosamente",
                extra={
                    "task_id": task_id,
                    "tarea": tarea_pipeline,
                    "dataset_name": dataset_name
                }
            )

            mapping = {
                "clasificacion": [1.0, 0.0, 0.0],
                "regresion":     [0.0, 1.0, 0.0],
                "clustering":    [0.0, 0.0, 1.0],
            }

            meta_features_vectorizadas.extend(mapping.get(tarea_pipeline, [0.0, 0.0, 0.0]))


            for num_pipeline in range (3):

                if tarea_pipeline == "clustering":
                    result_pipeline = minero.pipeline_no_supervisado(X, y, descripcion)
                else:
                    result_pipeline = minero.pipeline_supervisado(X, y, tarea_pipeline, descripcion)

                if result_pipeline.is_failure:
                    logger.error(
                        "Pipeline supervisado fallido", 
                        extra={
                            "task_id": task_id, 
                            "tarea": tarea_pipeline, 
                            "dataset_name": dataset_name,
                            "error": result_pipeline.get_error()
                        }
                    )
                    continue

                datos_pipeline = result_pipeline.get_value()
                pipeline = datos_pipeline["pipeline"]
                metricas = datos_pipeline["metricas"]
                lista_modelos_ml = datos_pipeline["modelos"]
                tiempos = datos_pipeline["tiempos"]

                logger.info(
                    "Contruccion de pipeline finalizada",
                    extra={
                        "task_id": task_id,
                        "tarea": tarea_pipeline,
                        "dataset_name": dataset_name,
                        "exitoso": True if metricas is not None else False,
                        "pipeline": pipeline,
                        "metricas": metricas,
                        "lista_modelos_ml": lista_modelos_ml,
                    }
                )

                for num_modelo, modelo in enumerate(lista_modelos_ml):
                    pipeline_vectorizado = vectorizar_pipeline(
                        mapa_indices, 
                        dimensiones, 
                        tarea_pipeline, 
                        pipeline, 
                        modelo
                    )
                    
                    for paso_t, pipeline_step in enumerate(pipeline_vectorizado):
                        vector_actual = meta_features_vectorizadas + pipeline_vectorizado[paso_t] 
                        vector_siguiente = meta_features_vectorizadas + pipeline_vectorizado[paso_t + 1] if paso_t + 1 < len(pipeline_vectorizado) else None
                        
                        fase_actual = list(pipeline.keys())[paso_t] if paso_t < len(pipeline) else "FINAL"
                        accion_actual = pipeline[fase_actual] if paso_t < len(pipeline) else ""
                        fase_accion = fase_actual + "_" + str(accion_actual)
                        pipeline_info = {
                            "nombre_dataset": dataset_name,
                            "num_pipeline": num_pipeline,
                            "num_modelo": num_modelo,
                            "mtf_json": json.dumps(meta_features),
                            "pipeline_json": json.dumps(pipeline),
                            "paso_t": paso_t,
                            "estado_actual": vector_actual,
                            "accion": fase_accion,
                            "estado_siguiente": vector_siguiente,
                            "nombre_modelo": modelo,
                            "tipo_tarea": tarea_pipeline,
                            "metricas": None if paso_t + 1 < len(pipeline_vectorizado) else json.dumps(metricas[num_modelo]),
                            "completado": 1 if paso_t + 1 == len(pipeline_vectorizado) else 0,
                            "tiempo_ejecucion": tiempos[num_modelo] if paso_t + 1 == len(pipeline_vectorizado) else None
                        }

                        db.guardar_resultados_pipeline(pipeline_info)


        
            print(f"Dataset: {dataset_name}")
            print(f"Task ID: {task_id}")
            print("Pipeline supervisado finalizado.")

if __name__ == "__main__":
    main()