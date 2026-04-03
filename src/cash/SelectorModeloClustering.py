import ast
import pandas as pd

from multiprocessing import Process, Queue

from ..LLM import LLM
from ..RegistroTecnica import RegistroTecnica
from ..ExtractorMetaFeatures import ExtractorMetaFeatures
from ..Result import Result

# K-Means
from sklearn.cluster import KMeans

# DBSCAN
from sklearn.cluster import DBSCAN

# Agglomerative Clustering (jerárquico)
from sklearn.cluster import AgglomerativeClustering

# Mean Shift
from sklearn.cluster import MeanShift

# Spectral Clustering
from sklearn.cluster import SpectralClustering

# Birch
from sklearn.cluster import Birch

class SelectorModeloClustering(RegistroTecnica):
    def __init__(self, random_state=None, config_test=None):
        super().__init__(log_fase="selector_modelo_clustering")
        self.random_state = random_state
        self.config_test = config_test
        self.llm_seleccionado = None
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
            self._calcular_parametros(X)

        self.registrar_algoritmo(self.log_algoritmo)
        return self
    
    def fit_model(self, modelo, X, queue):
        try:
            modelo.fit(X)
            queue.put(("ok", modelo))
        except Exception as e:
            queue.put(("fail", str(e)))
        
    def entrenar_modelo(self, X: pd.DataFrame) -> Result[object, str]:
        """
        Entrena una nueva instancia del modelo seleccionado con los hiperparámetros
        configurados. Si el entrenamiento excede el tiempo límite, se lanza una excepción.
        """
        modelo = self._get_instancia_modelo()
        hiper_parametros = self.log_params["params"]
        modelo.set_params(**hiper_parametros)

        queue = Queue()
        p = Process(target=self.fit_model, args=(modelo, X, queue))
        p.start()
        p.join(timeout=5)

        if p.is_alive():
            p.terminate()
            return Result.fail("Timeout: el entrenamiento excedio el limite de tiempo")
        
        status, result = queue.get()
        if status == "ok":
            return Result.ok(result)
        else:
            return Result.fail(result)
    
    def _get_instancia_modelo(self):
        match self.log_algoritmo:
            case "kmeans": return KMeans()
            case "dbscan": return DBSCAN()
            case "agglomerative_clustering": return AgglomerativeClustering()
            case "mean_shift": return MeanShift()
            case "spectral_clustering": return SpectralClustering()
            case "birch": return Birch()
            case _: raise ValueError(f"Modelo no reconocido: {self.log_algoritmo}")
        
    def _calcular_parametros(self, X: pd.DataFrame):
        llm = LLM(self.llm_seleccionado)

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
        