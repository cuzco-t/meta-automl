import ast
import numpy as np
import pandas as pd

from ..LLM import LLM
from ..RegistroTecnica import RegistroTecnica
from ..ExtractorMetaFeatures import ExtractorMetaFeatures

# Modelos lineales
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

# Support Vector Regression
from sklearn.svm import SVR

# KNN
from sklearn.neighbors import KNeighborsRegressor

# Árboles y ensembles
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

# Red neuronal
from sklearn.neural_network import MLPRegressor


class SelectorModeloRegresion(RegistroTecnica):
    def __init__(self, random_state=None, config_test=None):
        super().__init__(log_fase="selector_modelo")
        self.random_state = random_state
        self.config_test = config_test
        self.reiniciar()

    def reiniciar(self):
        self.log_algoritmo = None
        self.log_params = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Selecciona aleatoriamente el modelo de regresión a usar, y configura sus
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
                "lineal", 
                "ridge", 
                "lasso",
                "elasticnet",
                "svr",
                "knn",
                "arbol_decision",
                "random_forest",
                "gradient_boosting",
                "ada_boost",
                "mlp_regressor",
            ]
            self.log_algoritmo = generador_aleatorio.choice(MODELOS)

            self.registrar_algoritmo(self.log_algoritmo)
            # self._calcular_parametros(X, y)
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
        if self.log_algoritmo == "lineal":
            return LinearRegression()
        elif self.log_algoritmo == "ridge":
            return Ridge()
        elif self.log_algoritmo == "lasso":
            return Lasso()
        elif self.log_algoritmo == "elasticnet":
            return ElasticNet()
        elif self.log_algoritmo == "svr":
            return SVR()
        elif self.log_algoritmo == "knn":
            return KNeighborsRegressor()
        elif self.log_algoritmo == "arbol_decision":
            return DecisionTreeRegressor()
        elif self.log_algoritmo == "random_forest":
            return RandomForestRegressor()
        elif self.log_algoritmo == "gradient_boosting":
            return GradientBoostingRegressor()
        elif self.log_algoritmo == "ada_boost":
            return AdaBoostRegressor()
        elif self.log_algoritmo == "mlp_regressor":
            return MLPRegressor()
        else:
            raise ValueError(f"Modelo no reconocido: {self.log_algoritmo}")
        
    def _calcular_parametros(self, X: pd.DataFrame, y: pd.Series):
        llm = LLM("deepseek-r1:8b")

        extractor = ExtractorMetaFeatures()
        meta_features_globales_totales, _ = extractor.extraer_desde_dataframe(X.copy(), y.copy())
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
        