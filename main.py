from pathlib import Path

from src.PipelineLogger import PipelineLogger
from src.openml_descargador import OpenMLDescargador
from src.ExtractorMetaFeatures import ExtractorMetaFeatures
from src.minero.MineroDePipelines import MineroDePipelines
from src.minero.ejecutor_preprocesamiento import EjecutorPreprocesamiento
from src.vectorizador_pipeline import VectorizadorPipeline
from src.registrador_pipeline import RegistradorPipeline
from src.BaseDeDatos import BaseDeDatos
from src.orquestador_experimentos import OrquestadorExperimentos
from setproctitle import setproctitle
from src.config.Configuracion import Configuracion

import warnings
warnings.filterwarnings("ignore")

setproctitle("meta_automl")

def main():
    RUTA_DATASETS_CSV = "./data/datasets.csv"

    # Configurar logger
    logger = PipelineLogger().get_logger()

    # --- Dependencias ---
    loader = OpenMLDescargador()
    extractor = ExtractorMetaFeatures()
    minero = MineroDePipelines()
    db = BaseDeDatos()
    config = Configuracion()

    # Construir mapa de índices (necesita las fases y los modelos por tarea)
    ejecutor = EjecutorPreprocesamiento(None)
    fases_instancias = ejecutor.crear_fases_instancias()  # Corregir typo en el nombre del método si es necesario
    # Obtener modelos por tarea desde minero (suponiendo que existe el atributo)
    modelos_por_tarea = minero.tarea_modelos  # O podrías obtenerlos de los selectores directamente

    vectorizador = VectorizadorPipeline(fases_instancias, modelos_por_tarea)
    recorder = RegistradorPipeline(db, vectorizador)

    orquestador = OrquestadorExperimentos(
        loader=loader,
        extractor=extractor,
        minero=minero,
        vectorizador=vectorizador,
        recorder=recorder,
        logger=None,
        num_pipelines_por_dataset=config.num_pipelines_por_dataset
    )

    # Procesar datasets desde CSV (tarea, task_id)
    orquestador.ejecutar_csv(Path(RUTA_DATASETS_CSV))

if __name__ == "__main__":
    main()