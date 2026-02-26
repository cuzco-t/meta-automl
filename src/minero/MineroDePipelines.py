import numpy as np

from ..config.Configuracion import Configuracion
from ..PipelineLogger import PipelineLogger
from ..Result import Result

from ..cash.SelectorModeloClasificacion import SelectorModeloClasificacion
from ..cash.SelectorModeloRegresion import SelectorModeloRegresion
from ..cash.SelectorModeloClustering import SelectorModeloClustering

from .segmentador import Segmentador
from .entrenador import Entrenador
from .generador_pipeline import GeneradorPipeline
from .ejecutor_preprocesamiento import EjecutorPreprocesamiento
from .evaluador_modelos import EvaluadorModelos



class MineroDePipelines:
    def __init__(self):
        config = Configuracion()
        self.semilla = config.semilla_aleatoria
        self.rng = np.random.default_rng(self.semilla)
        self.logger = PipelineLogger().get_logger()
        self.generador = GeneradorPipeline(self.semilla, config.permitir_none, config.permitir_llm)
        self.ejecutor = EjecutorPreprocesamiento(self.logger)
        self.evaluador = EvaluadorModelos()
        self.n_folds = 3
        self.n_modelos = 10  # podrían venir de config
        self.tarea_modelos = {
            "clasificacion": SelectorModeloClasificacion().ALGORITMOS,
            "regresion": SelectorModeloRegresion().ALGORITMOS,
            "clustering": SelectorModeloClustering(self.semilla).ALGORITMOS
        }

    def pipeline_supervisado(self, X_df, y_df, tarea, descripcion=None):
        pipeline = self.generador.generar_pipeline_aleatorio(
            self.ejecutor.crear_fases_instancias()
        )
        lista_modelos = self.generador.generar_lista_modelos(self.n_modelos, self.tarea_modelos[tarea])
        self.logger.info("Pipeline generado", extra={"pipeline": pipeline, "modelos": lista_modelos})

        
        segmentador = Segmentador(n_splits=self.n_folds, random_state=self.semilla)
        folds = segmentador.segmentar(X_df, y_df, tipo_problema=tarea)
        self.logger.info(f"{self.n_folds} folds generados para tarea {tarea}")


        result_folds, tiempo_preprocesamiento = self.ejecutor.ejecutar_pipeline(folds, pipeline, tarea, descripcion)
        if result_folds.is_failure:
            self.logger.error("Error en ejecución del pipeline", extra={"error": result_folds.get_error()})
            return Result.fail(f"Error en ejecución del pipeline: {result_folds.get_error()}")

        folds_preprocesados = result_folds.get_value()
        self.logger.info("Pipeline ejecutado en todos los folds exitosamente")


        entrenador = Entrenador()
        results_modelos, tiempos_modelos = entrenador.entrenar(folds_preprocesados, lista_modelos, tarea)
        self.logger.info("Todos los modelos han sido entrenados en todos los folds")


        metricas_evaluacion = self.evaluador.evaluar_modelos(results_modelos, folds_preprocesados, tarea)
        tiempos_totales = [tiempo_preprocesamiento + t for t in tiempos_modelos]
        
        return Result.ok({
            "pipeline": pipeline,
            "modelos": lista_modelos,
            "metricas": metricas_evaluacion,
            "tiempos": tiempos_totales
        })

    def pipeline_no_supervisado(self, X_df, y_df=None, descripcion=None):
        # similar, pero llama a evaluador.evaluar_no_supervisado
        pipeline = self.generador.generar_pipeline_aleatorio(
            self.ejecutor.crear_fases_instancias()
        )
        lista_modelos = self.generador.generar_lista_modelos(self.n_modelos, self.tarea_modelos["clustering"])
        self.logger.info("Pipeline generado", extra={"pipeline": pipeline, "modelos": lista_modelos})

        result_datos, tiempo_preprocesamiento = self.ejecutor.ejecutar_pipeline_clustering(X_df, y_df, pipeline, descripcion)
        if result_datos.is_failure:
            self.logger.error("Error en ejecución del pipeline", extra={"error": result_datos.get_error()})
            datos_fallidos = result_datos.get_error()
            return Result.fail({
                "error": datos_fallidos['error'],
                "pipeline": pipeline,
                "fase": datos_fallidos["fase"]
            })
        
        datos = result_datos.get_value()
        X_proc = datos["X_proc"]
        y_proc = datos["y_proc"]


        entrenador = Entrenador()
        results_etiquetas, tiempos_modelos = entrenador.entrenar_clustering(X_proc, lista_modelos)
        self.logger.info("Todos los modelos han sido entrenados en todos los folds")

        metricas_evaluacion = self.evaluador.evaluar_modelos_clustering(results_etiquetas, X_proc, y_proc)
        tiempos_totales = [tiempo_preprocesamiento + t for t in tiempos_modelos]

        return Result.ok({
            "pipeline": pipeline,
            "modelos": lista_modelos,
            "metricas": metricas_evaluacion,
            "tiempos": tiempos_totales
        })
