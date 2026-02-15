import ast
import numpy as np
import pandas as pd

from ..LLM import LLM
from ..RegistroTecnica import RegistroTecnica
from ..ExtractorMetaFeatures import ExtractorMetaFeatures
from .SelectorHiperParametros import SelectorHiperParametros

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
    def __init__(self, random_state=None):
        super().__init__()
        self.log_fase = "selector_modelo"
        self.random_state = random_state
        self.modelo_seleccionado_ = None
        
    def get_modelo_ml(self, X: pd.DataFrame, y: pd.Series):
        # Si ya se ha seleccionado un modelo previamente, se devuelve una nueva
        # instancia de ese modelo con los mismos hiperparámetros
        if self.modelo_seleccionado_ is not None:
            modelo = self._get_instancia_modelo()
            modelo.set_params(**self.log_params)
            return modelo


        generador_aleatorio = np.random.default_rng(self.random_state)
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

        self.modelo_seleccionado_ = generador_aleatorio.choice(MODELOS)
        self.modelo_seleccionado_ = self._get_instancia_modelo()

        # llm = LLM("deepseek-r1:8b")

        # extractor = ExtractorMetaFeatures()
        # meta_features_globales = extractor.extraer(X, y)

        # prompt = llm.plantillas_prompts(
        #     plantilla="seleccionar_hiper_parametros",
        #     kwargs={
        #         "tarea": "regresión",
        #         "modelo_ml": self.modelo_seleccionado_,
        #         "meta_features_globales": meta_features_globales,
        #         "hiper_parametros_por_defecto": self.modelo_seleccionado_.get_params()
        #     }
        # )

        # hiper_parametros_texto = llm.generar_respuesta(prompt)
        # hiper_parametros = ast.literal_eval(hiper_parametros_texto)
        #! (Comentar en produccion) Esta linea es para probar el modelo sin usar el LLM
        hiper_parametros = self.modelo_seleccionado_.get_params()
        self.log_params = hiper_parametros
        self.registrar_tecnica(self.log_fase, self.modelo_seleccionado_.__class__.__name__, self.log_params)
        
        self.modelo_seleccionado_.set_params(**hiper_parametros)

        return self.modelo_seleccionado_

    def reiniciar(self):
        self.modelo_seleccionado_ = None
        self.log_params = {}
    
    def _get_instancia_modelo(self):
        if self.modelo_seleccionado_ == "lineal":
            return LinearRegression()
        elif self.modelo_seleccionado_ == "ridge":
            return Ridge()
        elif self.modelo_seleccionado_ == "lasso":
            return Lasso()
        elif self.modelo_seleccionado_ == "elasticnet":
            return ElasticNet()
        elif self.modelo_seleccionado_ == "svr":
            return SVR()
        elif self.modelo_seleccionado_ == "knn":
            return KNeighborsRegressor()
        elif self.modelo_seleccionado_ == "arbol_decision":
            return DecisionTreeRegressor()
        elif self.modelo_seleccionado_ == "random_forest":
            return RandomForestRegressor()
        elif self.modelo_seleccionado_ == "gradient_boosting":
            return GradientBoostingRegressor()
        elif self.modelo_seleccionado_ == "ada_boost":
            return AdaBoostRegressor()
        elif self.modelo_seleccionado_ == "mlp_regressor":
            return MLPRegressor()
        else:
            raise ValueError(f"Modelo no reconocido: {self.modelo_seleccionado_}")