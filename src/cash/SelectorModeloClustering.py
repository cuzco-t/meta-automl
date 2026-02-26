import ast
import signal
from importlib import import_module
import pandas as pd

from ..LLM import LLM
from ..RegistroTecnica import RegistroTecnica
from ..ExtractorMetaFeatures import ExtractorMetaFeatures
from ..Result import Result

class SelectorModeloClustering(RegistroTecnica):
    def __init__(self, random_state=None, config_test=None):
        super().__init__(log_fase="selector_modelo_clustering")
        self.random_state = random_state
        self.config_test = config_test
        self.ALGORITMOS = [
            "kmeans",
            "dbscan",
            "agglomerative_clustering",
            "mean_shift",
            "spectral_clustering",
            "birch"
        ]

    def calcular_hiper_parametros(self, X: pd.DataFrame):
        """
        Selecciona aleatoriamente el modelo de clustering a usar, y configura sus
        hiperparámetros.
        """
        if self.config_test is not None:
            self.log_algoritmo = self.config_test.get("algoritmo")
            self.log_params = self.config_test.get("params")

        else:
            self.registrar_algoritmo(self.log_algoritmo)
            # self._calcular_parametros(X)
            #! Comentar en produccion
            self.log_params["params"] = self._get_instancia_modelo().get_params()
            self.registrar_parametros(self.log_params)

        self.registrar_algoritmo(self.log_algoritmo)
        return self
    
    @staticmethod
    def timeout_handler(signum, frame):
        raise TimeoutError("Timeout: el entrenamiento excedio el limite de tiempo")
        
    def entrenar_modelo(self, X: pd.DataFrame) -> Result[object, str]:
        """
        Entrena una nueva instancia del modelo seleccionado con los hiperparámetros
        configurados. Si el entrenamiento excede el tiempo límite, se lanza una excepción.
        """
        modelo = self._get_instancia_modelo()
        hiper_parametros = self.log_params["params"]
        modelo.set_params(**hiper_parametros)

        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(5)
        
        result_entrenamiento = None
        try:
            etiquetas = modelo.fit_predict(X)
        except TimeoutError as e:
            result_entrenamiento = Result.fail(str(e))
        except Exception as e:
            result_entrenamiento = Result.fail(f"Error durante entrenamiento:\n{str(e)}")
        else:
            result_entrenamiento = Result.ok(etiquetas)
        finally:
            signal.alarm(0)

        return result_entrenamiento

    def _get_modelo_alias(self):
        """
        Mantiene compatibilidad de nombres históricos del pipeline,
        mapeándolos a modelos disponibles en cuML.
        """
        try:
            cluster = import_module("cuml.cluster")
        except ImportError as error:
            raise ImportError(
                "No se pudo importar cuML. Instala RAPIDS/cuML para usar los selectores GPU."
            ) from error

        KMeans = cluster.KMeans
        DBSCAN = cluster.DBSCAN
        AgglomerativeClustering = cluster.AgglomerativeClustering

        return {
            "kmeans": lambda: KMeans(),
            "dbscan": lambda: DBSCAN(),
            "agglomerative_clustering": lambda: AgglomerativeClustering(),
            "mean_shift": lambda: KMeans(),
            "spectral_clustering": lambda: KMeans(),
            "birch": lambda: KMeans(),
        }
    
    def _get_instancia_modelo(self):
        alias_modelos = self._get_modelo_alias()
        if self.log_algoritmo not in alias_modelos:
            raise ValueError(f"Modelo no reconocido: {self.log_algoritmo}")

        return alias_modelos[self.log_algoritmo]()
        
    def _calcular_parametros(self, X: pd.DataFrame):
        llm = LLM()

        extractor = ExtractorMetaFeatures()
        meta_features_globales_totales, _ = extractor.extraer_desde_dataframe(X.copy(), None)
        meta_features_globales_limpias = extractor.eliminar_constantes_errores(meta_features_globales_totales)
        meta_features_globales_formateadas = extractor.formatear_meta_features_globales(meta_features_globales_limpias)

        prompt = llm.plantillas_prompts(
            plantilla="seleccionar_hiper_parametros",
            kwargs={
                "tarea": "clustering",
                "modelo_ml": self.log_algoritmo,
                "meta_features_globales": meta_features_globales_formateadas,
                "hiper_parametros_por_defecto": self._get_instancia_modelo().get_params(),
            }
        )

        hiper_parametros_texto = llm.generar_respuesta(prompt)
        hiper_parametros = ast.literal_eval(hiper_parametros_texto)
        
        self.log_params["params"] = hiper_parametros
        self.registrar_parametros(self.log_params)
        