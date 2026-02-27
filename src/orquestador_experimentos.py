import logging
import pandas as pd

from pathlib import Path
from typing import Iterator

from src.config.Configuracion import Configuracion
from src.openml_descargador import OpenMLDescargador
from src.ExtractorMetaFeatures import ExtractorMetaFeatures
from src.minero.MineroDePipelines import MineroDePipelines
from src.vectorizador_pipeline import VectorizadorPipeline
from src.registrador_pipeline import RegistradorPipeline


class OrquestadorExperimentos:
    """
    Orquesta la ejecución de experimentos:
    - Lee archivos con listas de task_ids.
    - Para cada task_id, descarga datos, extrae metafeatures y ejecuta pipelines.
    - Registra resultados en base de datos.
    """

    def __init__(
        self,
        loader: OpenMLDescargador,
        extractor: ExtractorMetaFeatures,
        minero: MineroDePipelines,
        vectorizador: VectorizadorPipeline,
        recorder: RegistradorPipeline,
        logger: logging.Logger,
        num_pipelines_por_dataset: int = 3
    ):
        config = Configuracion()

        self.loader = loader
        self.extractor = extractor
        self.minero = minero
        self.vectorizador = vectorizador
        self.recorder = recorder
        self.logger = logger
        self.num_pipelines = config.num_pipelines_por_dataset

        # Mapeo de tarea a vector one-hot
        self.tarea_onehot = {
            "clasificacion": [1.0, 0.0, 0.0],
            "regresion": [0.0, 1.0, 0.0],
            "clustering": [0.0, 0.0, 1.0],
        }

    def ejecutar_archivo(self, ruta_archivo: Path) -> None:
        """Procesa un archivo que contiene una lista de task_ids (uno por línea)."""
        tarea = ruta_archivo.stem.split("_")[0]  # ej: "clasificacion_task_ids.txt" -> "clasificacion"
        self.logger.info(f"Procesando archivo {ruta_archivo.name}, tarea={tarea}")

        for task_id in self._leer_task_ids(ruta_archivo):
            self._procesar_task(task_id, tarea)

    def _leer_task_ids(self, ruta: Path) -> Iterator[int]:
        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if linea:
                    yield int(linea)

    def _procesar_task(self, task_id: int, tarea: str) -> None:
        """Procesa un único task_id: descarga, extrae metafeatures, ejecuta pipelines."""
        # 1. Descargar datos
        result_datos = self.loader.obtener_datos_tarea(task_id)
        if result_datos.is_failure:
            self.logger.error(f"Error descargando task {task_id}: {result_datos.get_error()}")
            return

        dataset_name, descripcion, X, y = result_datos.get_value()
        self.logger.info(f"Dataset {dataset_name} (task {task_id}) descargado correctamente.")

        # 2. Extraer metafeatures
        meta_features, meta_features_vector = self.extractor.extraer_desde_dataframe(
            X, y, vectorizar=True
        )
        # Añadir one-hot de tarea
        meta_features_vector.extend(self.tarea_onehot.get(tarea, [0.0, 0.0, 0.0]))

        # 3. Ejecutar N pipelines aleatorios
        for num_pipeline in range(1, self.num_pipelines + 1):
            self._ejecutar_pipeline(
                num_pipeline,
                dataset_name,
                descripcion,
                X,
                y,
                tarea,
                meta_features,
                meta_features_vector,
            )
        
        self.recorder.flush()

    def _ejecutar_pipeline(
        self,
        num_pipeline: int,
        dataset_name: str,
        descripcion: str,
        X: pd.DataFrame,
        y: pd.Series,
        tarea: str,
        meta_features: dict,
        meta_features_vector: list,
    ) -> None:
        """Ejecuta un pipeline (supervisado o no supervisado) y registra los resultados."""
        # Llamar al método correspondiente de MineroDePipelines
        if tarea == "clustering":
            result = self.minero.pipeline_no_supervisado(X, y, descripcion)
        else:
            if dataset_name == "breast-cancer":
                print("DEBUG: Ejecutando pipeline supervisado con tarea =", tarea)
            result = self.minero.pipeline_supervisado(X, y, tarea, descripcion)

        if result.is_failure:
            datos_fallidos = result.get_error()
            self.recorder.guardar_ejecucion_con_fallo(
                dataset_name=dataset_name,
                tarea=tarea,
                num_pipeline=num_pipeline,
                meta_features=meta_features,
                meta_features_vector=meta_features_vector,
                pipeline=datos_fallidos["pipeline"],
                fase=datos_fallidos["fase"],
                error=datos_fallidos["error"]
            )
            return

        datos = result.get_value()
        pipeline = datos["pipeline"]
        metricas = datos["metricas"]      # dict con listas de métricas por modelo
        modelos = datos["modelos"]        # lista de nombres de modelos
        tiempos = datos["tiempos"]        # lista de tiempos por modelo

        self.recorder.guardar_ejecucion(
            dataset_name=dataset_name,
            tarea=tarea,
            num_pipeline=num_pipeline,
            meta_features=meta_features,
            meta_features_vector=meta_features_vector,
            pipeline=pipeline,
            modelos=modelos,
            metricas_por_modelo=metricas,
            tiempos=tiempos,
        )
        