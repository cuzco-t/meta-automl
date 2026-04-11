import json
import numpy as np
from dataclasses import dataclass

from ..config.Configuracion import Configuracion
from ..Result import Result

from ..cash.SelectorModeloClasificacion import SelectorModeloClasificacion
from ..cash.SelectorModeloRegresion import SelectorModeloRegresion
from ..cash.SelectorModeloClustering import SelectorModeloClustering

from .segmentador import Segmentador
from .entrenador import Entrenador
from .generador_pipeline import GeneradorPipeline
from .ejecutor_preprocesamiento import EjecutorPreprocesamiento
from .evaluador_modelos import EvaluadorModelos

from datetime import datetime
print_original = print

def print(*args, **kwargs):
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_original(f"{ahora} |", *args, **kwargs)


@dataclass(frozen=True)
class ConfiguracionPipeline:
    llm_seleccionado: str | None
    pipeline: dict[str, str | None]
    lista_modelos: list[str]

class MineroDePipelines:
    def __init__(self):
        config = Configuracion()
        self.semilla = config.semilla_aleatoria
        self.rng = np.random.default_rng(self.semilla)
        self.generador = GeneradorPipeline(self.semilla, config.permitir_none, config.permitir_llm)
        self.ejecutor = EjecutorPreprocesamiento(self.logger)
        self.evaluador = EvaluadorModelos()
        self.n_folds = 3
        self.n_modelos = config.num_modelos_por_pipeline
        self.tarea_modelos = {
            "clasificacion": SelectorModeloClasificacion().ALGORITMOS,
            "regresion": SelectorModeloRegresion().ALGORITMOS,
            "clustering": SelectorModeloClustering(self.semilla).ALGORITMOS
        }

    def _seleccionar_llm(self):
        opciones = [
            "deepseek-r1:8b",
            "llama3.1:8b",
            "qwen2.5-coder:7b",
            None  # Opcion para usar hiperparámetros por defecto sin LLM
        ]
        return self.rng.choice(opciones)

    def _vectorizar_llm_seleccionado(self, llm):
        mapping = {
            "deepseek-r1:8b": [1.0, 0.0, 0.0, 0.0],
            "llama3.1:8b": [0.0, 1.0, 0.0, 0.0],
            "qwen2.5-coder:7b": [0.0, 0.0, 1.0, 0.0],
            None: [0.0, 0.0, 0.0, 1.0]
        }
        return mapping.get(llm, [0.0, 0.0, 0.0, 1.0])

    def preparar_configuracion_pipeline(self, tarea: str) -> ConfiguracionPipeline:
        """Genera la configuración completa de un pipeline antes de ejecutarlo."""
        return ConfiguracionPipeline(
            llm_seleccionado=self._seleccionar_llm(),
            pipeline=self.generador.generar_pipeline_aleatorio(
                self.ejecutor.crear_fases_instancias()
            ),
            lista_modelos=self._get_lista_modelos(tarea),
        )

    def preparar_configuraciones_pipeline(self, tarea: str, cantidad: int) -> list[ConfiguracionPipeline]:
        """Genera en orden todas las configuraciones que luego pueden ejecutarse en paralelo."""
        return [self.preparar_configuracion_pipeline(tarea) for _ in range(cantidad)]

    def ejecutar_pipeline_configurado(
        self,
        X_df,
        y_df,
        tarea: str,
        configuracion: ConfiguracionPipeline,
        descripcion=None,
    ):
        """Ejecuta un pipeline ya configurado, manteniendo la lógica original de entrenamiento."""
        llm_seleccionado = configuracion.llm_seleccionado
        pipeline = configuracion.pipeline
        lista_modelos = configuracion.lista_modelos

        print("-" * 50)
        print("Pipeline generado")
        print("-" * 50)
        print(json.dumps(pipeline, indent=4))

        print("-" * 50)
        print("Modelos seleccionados")
        print("-" * 50)
        [print(f"{i+1}. {modelo}") for i, modelo in enumerate(lista_modelos)]
        print("LLM seleccionado:", llm_seleccionado)
        print("-" * 50)

        segmentador = Segmentador(n_splits=self.n_folds, random_state=self.semilla)
        folds = segmentador.segmentar(X_df, y_df, tipo_problema=tarea)

        print("OK - Folds generados para tarea:", tarea)

        result_folds, tiempo_preprocesamiento = self.ejecutor.ejecutar_pipeline(folds, pipeline, tarea, descripcion)
        if result_folds.is_failure:
            datos_fallidos = result_folds.get_error()
            return Result.fail({
                "error": datos_fallidos['error'],
                "pipeline": pipeline,
                "fase": datos_fallidos["fase"],
                "llm_seleccionado": llm_seleccionado,
                "llm_vector": self._vectorizar_llm_seleccionado(llm_seleccionado)
            })

        folds_preprocesados = result_folds.get_value()
        print("OK - Pipeline ejecutado en todos los folds exitosamente")

        print("-" * 50)
        print(f"Entrenando modelos para tarea '{tarea}'")
        print("-" * 50)
        entrenador = Entrenador()
        results_modelos, tiempos_modelos = entrenador.entrenar(
            folds_preprocesados,
            lista_modelos,
            tarea,
            llm_seleccionado
        )
        print("OK - Todos los modelos han sido entrenados en todos los folds")

        metricas_evaluacion = self.evaluador.evaluar_modelos(results_modelos, folds_preprocesados, tarea)
        tiempos_totales = [tiempo_preprocesamiento + t for t in tiempos_modelos]

        return Result.ok({
            "pipeline": pipeline,
            "modelos": lista_modelos,
            "metricas": metricas_evaluacion,
            "tiempos": tiempos_totales,
            "llm_seleccionado": llm_seleccionado,
            "llm_vector": self._vectorizar_llm_seleccionado(llm_seleccionado)
        })

    def ejecutar_pipeline_no_supervisado_configurado(
        self,
        X_df,
        y_df=None,
        configuracion: ConfiguracionPipeline | None = None,
        descripcion=None,
    ):
        """Ejecuta un pipeline de clustering ya configurado."""
        if configuracion is None:
            configuracion = self.preparar_configuracion_pipeline("clustering")

        llm_seleccionado = configuracion.llm_seleccionado
        pipeline = configuracion.pipeline
        lista_modelos = configuracion.lista_modelos

        print("-" * 50)
        print("Pipeline generado")
        print("-" * 50)
        print(json.dumps(pipeline, indent=4))

        print("-" * 50)
        print("Modelos seleccionados")
        print("-" * 50)
        [print(f"{i+1}. {modelo}") for i, modelo in enumerate(lista_modelos)]
        print("LLM seleccionado:", llm_seleccionado)
        print("-" * 50)

        result_datos, tiempo_preprocesamiento = self.ejecutor.ejecutar_pipeline_clustering(X_df, y_df, pipeline, descripcion)

        if result_datos.is_failure:
            datos_fallidos = result_datos.get_error()
            return Result.fail({
                "error": datos_fallidos['error'],
                "pipeline": pipeline,
                "fase": datos_fallidos["fase"],
                "llm_seleccionado": llm_seleccionado,
                "llm_vector": self._vectorizar_llm_seleccionado(llm_seleccionado)
            })

        datos = result_datos.get_value()
        X_proc = datos["X_proc"]
        y_proc = datos["y_proc"]

        print("-" * 50)
        print(f"Entrenando modelos para tarea 'clustering'")
        print("-" * 50)
        entrenador = Entrenador()
        results_modelos, tiempos_modelos = entrenador.entrenar_clustering(
            X_proc,
            lista_modelos,
            llm_seleccionado
        )

        print("OK - Todos los modelos han sido entrenados en todos los folds")

        metricas_evaluacion = self.evaluador.evaluar_modelos_clustering(results_modelos, X_proc, y_proc)
        tiempos_totales = [tiempo_preprocesamiento + t for t in tiempos_modelos]

        return Result.ok({
            "pipeline": pipeline,
            "modelos": lista_modelos,
            "metricas": metricas_evaluacion,
            "tiempos": tiempos_totales,
            "llm_seleccionado": llm_seleccionado,
            "llm_vector": self._vectorizar_llm_seleccionado(llm_seleccionado)
        })

    def pipeline_supervisado(self, X_df, y_df, tarea, descripcion=None):
        configuracion = self.preparar_configuracion_pipeline(tarea)
        return self.ejecutar_pipeline_configurado(X_df, y_df, tarea, configuracion, descripcion)

    def pipeline_no_supervisado(self, X_df, y_df=None, descripcion=None):
        configuracion = self.preparar_configuracion_pipeline("clustering")
        return self.ejecutar_pipeline_no_supervisado_configurado(X_df, y_df, configuracion, descripcion)

    def _get_lista_modelos(self, tarea):
        # Obtener la lista completa de modelos según la tarea
        if tarea == "clasificacion":
            modelos = SelectorModeloClasificacion().ALGORITMOS
        elif tarea == "regresion":
            modelos = SelectorModeloRegresion().ALGORITMOS
        elif tarea == "clustering":
            modelos = SelectorModeloClustering(self.semilla).ALGORITMOS
        else:
            raise ValueError(f"Tarea desconocida: {tarea}")

        # Calcular tamaño de la sublista (70% sin reemplazo)
        n = max(1, int(len(modelos) * 0.7))  # al menos 1 modelo
        sublista = list(self.rng.choice(modelos, size=n, replace=False))

        return sublista