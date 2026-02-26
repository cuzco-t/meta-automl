import json

from typing import Dict, List, Any

from src.BaseDeDatos import BaseDeDatos
from .vectorizador_pipeline import VectorizadorPipeline

class RegistradorPipeline:
    """Registra en la base de datos los resultados de cada ejecución."""

    def __init__(self, db: BaseDeDatos, vectorizador: VectorizadorPipeline):
        self.db = db
        self.vectorizador = vectorizador

    def guardar_ejecucion(
        self,
        dataset_name: str,
        tarea: str,
        num_pipeline: int,
        meta_features: Dict[str, Any],
        meta_features_vector: List[float],
        pipeline: Dict[str, str],
        modelos: List[str],
        metricas_por_modelo: List[Dict[str, float]],
        tiempos: List[float]
    ) -> None:
        """
        Para cada modelo asociado al pipeline, genera los vectores paso a paso
        y guarda un registro por cada paso en la base de datos.
        """
        for idx_modelo, modelo in enumerate(modelos):
            metricas_modelo = metricas_por_modelo[idx_modelo]
            historia = self.vectorizador.vectorizar_pipeline(tarea, pipeline, modelo)

            for paso_t, vector_estado in enumerate(historia):
                es_final = (paso_t == len(historia) - 1)
                vector_actual = meta_features_vector + vector_estado
                vector_siguiente = None
                if not es_final:
                    vector_siguiente = meta_features_vector + historia[paso_t + 1]

                # Determinar fase y acción actual
                if paso_t < len(pipeline):
                    fase_actual = list(pipeline.keys())[paso_t]
                    accion_actual = pipeline[fase_actual]
                    fase_accion = f"{fase_actual}_{accion_actual}"
                else:
                    fase_accion = "FINAL"

                registro = {
                    "nombre_dataset": dataset_name,
                    "num_pipeline": num_pipeline,
                    "num_modelo": idx_modelo + 1,
                    "mtf_json": json.dumps(meta_features),
                    "pipeline_json": json.dumps(pipeline),
                    "paso_t": paso_t,
                    "estado_actual": vector_actual,
                    "accion": fase_accion,
                    "estado_siguiente": vector_siguiente,
                    "nombre_modelo": modelo,
                    "tipo_tarea": tarea,
                    "metricas": None if not es_final else json.dumps(metricas_modelo),
                    "completado": 1 if es_final else 0,
                    "tiempo_ejecucion": tiempos[idx_modelo] if es_final else None,
                }

                self.db.guardar_resultados_pipeline(registro)
                