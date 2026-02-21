import ast
import signal
import time

import numpy as np
import pandas as pd

from ..LLM import LLM
from ..RegistroTecnica import RegistroTecnica
from ..ExtractorMetaFeatures import ExtractorMetaFeatures
from ..Result import Result

# Modelos lineales
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    RidgeClassifier
)

# Arboles y ensembles
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier
)

# SVM
from sklearn.svm import SVC, LinearSVC

# KNN
from sklearn.neighbors import KNeighborsClassifier

# Naive Bayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# Red Neural
from sklearn.neural_network import MLPClassifier

# Discriminantes
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)


class SelectorModeloClasificacion(RegistroTecnica):
    def __init__(self, config_test=None):
        super().__init__(log_fase="selector_modelo")
        self.config_test = config_test
        self.ALGORITMOS = [
            "logistic_regression",
            "sgd_classifier",
            "ridge_classifier",
            "decision_tree",
            "random_forest",
            "gradient_boosting",
            "hist_gradient_boosting",
            "ada_boost",
            "svc",
            "linear_svc",
            "knn",
            "gaussian_nb",
            "multinomial_nb",
            "mlp_classifier",
            "linear_discriminant_analysis",
            "quadratic_discriminant_analysis"
        ]
        
    def calcular_hiper_parametros(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Consulta al LLM para obtener los hiperparámetros recomendados para el modelo seleccionado,
        basándose en las meta-features del dataset.
        """        
        if self.config_test is not None:
            self.log_algoritmo = self.config_test.get("algoritmo")
            self.log_params = self.config_test.get("params")

        else:
            self.registrar_algoritmo(self.log_algoritmo)
            # self._calcular_parametros(X, y)
            #! Comentar en produccion
            self.log_params["params"] = self._get_instancia_modelo().get_params()
            self.registrar_parametros(self.log_params)

        self.registrar_algoritmo(self.log_algoritmo)
        return None

    @staticmethod
    def timeout_handler(signum, frame):
        raise TimeoutError("Timeout: el entrenamiento excedio el limite de tiempo")

    def entrenar_modelo(self, X: pd.DataFrame, y: pd.Series) -> Result[object, str]:
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
            modelo.fit(X, y)
        except TimeoutError as e:
            result_entrenamiento = Result.fail(str(e))
        except Exception as e:
            result_entrenamiento = Result.fail(f"Error durante entrenamiento:\n{str(e)}")
        else:
            result_entrenamiento = Result.ok(modelo)
        finally:
            signal.alarm(0)

        return result_entrenamiento
    
    def _get_instancia_modelo(self):
        match self.log_algoritmo:
            case "logistic_regression": return LogisticRegression()
            case "sgd_classifier": return SGDClassifier()
            case "ridge_classifier": return RidgeClassifier()
            case "decision_tree": return DecisionTreeClassifier()
            case "random_forest": return RandomForestClassifier()
            case "gradient_boosting": return GradientBoostingClassifier()
            case "hist_gradient_boosting": return HistGradientBoostingClassifier()
            case "ada_boost": return AdaBoostClassifier()
            case "svc": return SVC()
            case "linear_svc": return LinearSVC()
            case "knn": return KNeighborsClassifier()
            case "gaussian_nb": return GaussianNB()
            case "multinomial_nb": return MultinomialNB()
            case "mlp_classifier": return MLPClassifier()
            case "linear_discriminant_analysis": return LinearDiscriminantAnalysis()
            case "quadratic_discriminant_analysis": return QuadraticDiscriminantAnalysis()
            case _: raise ValueError(f"Modelo no reconocido: {self.log_algoritmo}")
        
    def _calcular_parametros(self, X: pd.DataFrame, y: pd.Series):
        llm = LLM("deepseek-r1:8b")

        extractor = ExtractorMetaFeatures()
        meta_features_globales_totales, _ = extractor.extraer_desde_dataframe(X.copy(), y.copy())
        meta_features_globales_limpias = extractor.eliminar_constantes_errores(meta_features_globales_totales)
        meta_features_globales_formateadas = extractor.formatear_meta_features_globales(meta_features_globales_limpias)

        prompt = llm.plantillas_prompts(
            plantilla="seleccionar_hiper_parametros",
            kwargs={
                "tarea": "clasificación",
                "modelo_ml": self.log_algoritmo,
                "meta_features_globales": meta_features_globales_formateadas,
                "hiper_parametros_por_defecto": self._get_instancia_modelo().get_params(),
            }
        )

        hiper_parametros_texto = llm.generar_respuesta(prompt)
        hiper_parametros = ast.literal_eval(hiper_parametros_texto)
        
        self.log_params["params"] = hiper_parametros
        self.registrar_parametros(self.log_params)
        