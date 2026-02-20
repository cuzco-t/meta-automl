import ast
import numpy as np
import pandas as pd

from ..LLM import LLM
from ..RegistroTecnica import RegistroTecnica
from ..ExtractorMetaFeatures import ExtractorMetaFeatures

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
        super().__init__(log_fase="selector_modelo")
        self.random_state = random_state
        self.config_test = config_test
        

    def reiniciar(self):
        self.log_algoritmo = None
        self.log_params = {}

    def fit(self, X: pd.DataFrame):
        """
        Selecciona aleatoriamente el modelo de clustering a usar, y configura sus
        hiperparámetros.
        """
        if self.log_algoritmo is not None:
            return self
        
        if self.config_test is not None:
            self.log_algoritmo = self.config_test.get("algoritmo")
            self.log_params = self.config_test.get("params")

        else:
            generador_aleatorio = np.random.default_rng()
            MODELOS = [
                "kmeans",
                "dbscan",
                "agglomerative_clustering",
                "mean_shift",
                "spectral_clustering",
                "birch"
            ]
            self.log_algoritmo = generador_aleatorio.choice(MODELOS)

            self.registrar_algoritmo(self.log_algoritmo)
            # self._calcular_parametros(X)
            #! Comentar en produccion
            self.log_params["params"] = self._get_instancia_modelo().get_params()
            self.registrar_parametros(self.log_params)

        self.registrar_algoritmo(self.log_algoritmo)
        return self
        
    def get_modelo_ml(self):
        modelo = self._get_instancia_modelo()
        hiper_parametros = self.log_params["params"]

        modelo.set_params(**hiper_parametros)

        return modelo
    
    def _get_instancia_modelo(self):
        if self.log_algoritmo == "kmeans":
            return KMeans()
        elif self.log_algoritmo == "dbscan":
            return DBSCAN()
        elif self.log_algoritmo == "agglomerative_clustering":
            return AgglomerativeClustering()
        elif self.log_algoritmo == "mean_shift":
            return MeanShift()
        elif self.log_algoritmo == "spectral_clustering":
            return SpectralClustering()
        elif self.log_algoritmo == "birch":
            return Birch()
        else:
            raise ValueError(f"Modelo no reconocido: {self.log_algoritmo}")
        
    def _calcular_parametros(self, X: pd.DataFrame):
        llm = LLM("deepseek-r1:8b")

        extractor = ExtractorMetaFeatures()
        meta_features_globales_totales, _ = extractor.extraer_desde_dataframe(X.copy(), None)
        meta_features_globales_limpias = extractor.eliminar_constantes_errores(meta_features_globales_totales)
        meta_features_globales_formateadas = extractor.formatear_meta_features_globales(meta_features_globales_limpias)

        prompt = llm.plantillas_prompts(
            plantilla="seleccionar_hiper_parametros",
            kwargs={
                "tarea": "regresión",
                "modelo_ml": self.log_algoritmo,
                "meta_features_globales": meta_features_globales_formateadas,
                "hiper_parametros_por_defecto": self._get_instancia_modelo().get_params(),
            }
        )

        hiper_parametros_texto = llm.generar_respuesta(prompt)
        hiper_parametros = ast.literal_eval(hiper_parametros_texto)
        
        self.log_params["params"] = hiper_parametros
        self.registrar_parametros(self.log_params)
        