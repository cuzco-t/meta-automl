import logging
import time
import pandas as pd
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue

from pathlib import Path
from typing import Iterator

from src.config.Configuracion import Configuracion
from src.openml_descargador import OpenMLDescargador
from src.ExtractorMetaFeatures import ExtractorMetaFeatures
from src.minero.MineroDePipelines import MineroDePipelines
from src.vectorizador_pipeline import VectorizadorPipeline
from src.registrador_pipeline import RegistradorPipeline
from src.Result import Result
from src.PipelineLogger import PipelineLogger

from datetime import datetime

print_original = print

def print(*args, **kwargs):
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_original(f"{ahora} |", *args, **kwargs)

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
        self.logger = PipelineLogger().get_logger() 
        self.num_pipelines = config.num_pipelines_por_dataset

        # Mapeo de tarea a vector one-hot
        self.tarea_onehot = {
            "clasificacion": [1.0, 0.0, 0.0],
            "regresion": [0.0, 1.0, 0.0],
            "clustering": [0.0, 0.0, 1.0],
        }

    def ejecutar_archivo(self, ruta_archivo: Path) -> None:
        """Procesa un archivo que contiene una lista de task_ids (uno por línea)."""
        tarea = self._normalizar_tarea(ruta_archivo.stem.split("_")[0])  # ej: "clasificacion_task_ids.txt" -> "clasificacion"
        
        print("=" * 50)
        print("PROCESANDO ARCHIVO:", ruta_archivo.name)
        print("=" * 50)

        for task_id in self._leer_task_ids(ruta_archivo):
            self._procesar_task(task_id, tarea)

    def ejecutar_csv(self, ruta_csv: Path) -> None:
        """Procesa un CSV con dos columnas: tarea y task_id."""
        df = pd.read_csv(ruta_csv, header=None, names=["tarea", "task_id"])

        for fila in df.itertuples(index=False):
            tarea = self._normalizar_tarea(fila.tarea)
            task_id = int(fila.task_id)
            self._procesar_task(task_id, tarea)

    def _leer_task_ids(self, ruta: Path) -> Iterator[int]:
        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if linea:
                    yield int(linea)

    def _pipeline_multiproceso(
        self, 
        num_pipeline: int, 
        X: pd.DataFrame, 
        y: pd.Series, 
        tarea: str, 
        descripcion: str, 
        configuracion,
        queue: Queue
    ) -> dict:
        try:
            resultado = self._ejecutar_pipeline(
                num_pipeline=num_pipeline,
                X=X,
                y=y,
                tarea=tarea,
                descripcion=descripcion,
                configuracion=configuracion,
            )
            queue.put((num_pipeline, "ok", resultado))
        except Exception as e:
            queue.put((num_pipeline, "fail", str(e)))

    def _procesar_task(self, task_id: int, tarea: str) -> None:
        """Procesa un único task_id: descarga, extrae metafeatures, ejecuta pipelines."""

        print("=" * 50)
        print(f"INICIANDO PROCESO PARA TASK_ID: {task_id} (Tarea: {tarea})")
        print("=" * 50)

        # 1. Descargar datos
        result_datos = self.loader.obtener_datos_tarea(task_id)
        if result_datos.is_failure:
            print(f"ERROR - Descargando task: {task_id}")
            return

        dataset_name, descripcion, X, y = result_datos.get_value()
        print("OK - Datos descargados para task_id:", task_id)

        # 2. Extraer metafeatures
        meta_features, meta_features_vector = self.extractor.extraer_desde_dataframe(
            X, y, vectorizar=True
        )
        print("OK - Metafeatures extraídas para task_id:", task_id)

        # Añadir one-hot de tarea
        meta_features_vector_tarea = meta_features_vector + self.tarea_onehot.get(tarea, [0.0, 0.0, 0.0])

        configuraciones = self.minero.preparar_configuraciones_pipeline(tarea, self.num_pipelines)

        cola_resultados = Queue()

        procesos = [
            Process(
                target=self._pipeline_multiproceso,
                args=(i+1, X, y, tarea, descripcion, config, cola_resultados)
            )
            for i, config in enumerate(configuraciones)
        ]

        for p in procesos:
            p.start()

        procesos_activos = len(procesos)
        start_time = time.time()
        timeout = 1200  # 20 minutos

        while procesos_activos > 0:
            try:
                # Espera no bloqueante larga pero controlada
                pipeline_id, status, result = cola_resultados.get(timeout=1)

                if status == "ok":
                    self.logger.info(f"Pipeline {pipeline_id} completado exitosamente.")

                    print("\n"*5)
                    print("="*100)
                    print("OK - Este pipeline se completo correctamente")
                    self._registrar_resultado_pipeline(
                        dataset_name=dataset_name,
                        tarea=tarea,
                        num_pipeline=pipeline_id,
                        meta_features=meta_features,
                        meta_features_vector=meta_features_vector_tarea,
                        result=result["result"],
                    )
                else:
                    self.logger.info(f"Pipeline {pipeline_id} falló.")

                procesos_activos -= 1

            except Exception as e:
                print("\n"*5)
                print(f"Error al procesar resultado: {e}")
                pass  # No hay resultados aún

            # Control de timeout global
            if time.time() - start_time > timeout:
                self.logger.info("Timeout global alcanzado. Terminando procesos...")
                print("\n")
                print("TIMEOUT GLOBAL")
                for i, p in enumerate(procesos):
                    if p.is_alive():
                        p.terminate()
                        self.logger.info(f"Pipeline {i+1} terminado por timeout.")
                break

        # Limpieza final
        for p in procesos:
            print("\n"*5)
            print("LIMPIEZA")
            p.join()
        

    def _normalizar_tarea(self, tarea: str) -> str:
        """Convierte tarea a formato interno: clasificacion, regresion o clustering."""
        texto = str(tarea).strip().lower()
        texto = "".join(
            c for c in unicodedata.normalize("NFD", texto)
            if unicodedata.category(c) != "Mn"
        )
        return texto

    def _ejecutar_pipelines_en_paralelo(
        self,
        dataset_name: str,
        descripcion: str,
        X: pd.DataFrame,
        y: pd.Series,
        tarea: str,
        meta_features: dict,
        meta_features_vector: list,
        configuraciones: list,
    ) -> list[dict]:
        """Ejecuta los pipelines en paralelo y devuelve sus resultados en orden."""
        if not configuraciones:
            return []

        max_workers = max(1, min(len(configuraciones), self.num_pipelines))
        resultados = []

        self.logger.info(f"Iniciando ejecución pipelines en paralelo (max_workers={max_workers}) para dataset '{dataset_name}'")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futuros = []
            for num_pipeline, configuracion in enumerate(configuraciones, start=1):
                futuros.append(
                    executor.submit(
                        self._ejecutar_pipeline,
                        num_pipeline,
                        X,
                        y,
                        tarea,
                        descripcion,
                        configuracion,
                    )
                )

            for future in futuros:
                future_result = future.result()  # Esto esperará a que cada pipeline termine
                self.logger.info(f"Pipeline {future_result['num_pipeline']} completado.")
                resultados.append(future_result)

        return resultados

    def _ejecutar_pipeline(
        self,
        num_pipeline: int,
        X: pd.DataFrame,
        y: pd.Series,
        tarea: str,
        descripcion: str,
        configuracion,
    ) -> dict:
        """Ejecuta un pipeline ya configurado y devuelve el resultado para persistirlo luego."""
        try:
            if tarea == "clustering":
                result = self.minero.ejecutar_pipeline_no_supervisado_configurado(
                    X,
                    y,
                    configuracion,
                    descripcion,
                )
            else:
                result = self.minero.ejecutar_pipeline_configurado(
                    X,
                    y,
                    tarea,
                    configuracion,
                    descripcion,
                )
        except Exception as exc:
            result = Result.fail({
                "error": str(exc),
                "pipeline": configuracion.pipeline,
                "fase": "desconocida",
                "llm_seleccionado": configuracion.llm_seleccionado,
                "llm_vector": self.minero._vectorizar_llm_seleccionado(configuracion.llm_seleccionado),
            })

        return {
            "num_pipeline": num_pipeline,
            "result": result,
        }

    def _registrar_resultado_pipeline(
        self,
        dataset_name: str,
        tarea: str,
        num_pipeline: int,
        meta_features: dict,
        meta_features_vector: list,
        result: Result,
    ) -> None:
        """Guarda en base de datos el resultado ya resuelto de un pipeline."""
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
                error=datos_fallidos["error"],
                llm_seleccionado=datos_fallidos["llm_seleccionado"],
                llm_vector=datos_fallidos["llm_vector"],
            )
            return

        datos = result.get_value()
        self.recorder.guardar_ejecucion(
            dataset_name=dataset_name,
            tarea=tarea,
            num_pipeline=num_pipeline,
            meta_features=meta_features,
            meta_features_vector=meta_features_vector,
            pipeline=datos["pipeline"],
            modelos=datos["modelos"],
            metricas_por_modelo=datos["metricas"],
            tiempos=datos["tiempos"],
            llm_seleccionado=datos["llm_seleccionado"],
            llm_vector=datos["llm_vector"],
        )
        
