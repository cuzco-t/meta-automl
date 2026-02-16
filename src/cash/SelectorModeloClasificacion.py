import ast
import numpy as np
import pandas as pd

from ..LLM import LLM
from ..RegistroTecnica import RegistroTecnica
from ..ExtractorMetaFeatures import ExtractorMetaFeatures

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
    def __init__(self, random_state=None, config_test=None):
        super().__init__(log_fase="selector_modelo")
        self.random_state = random_state
        self.config_test = config_test
        self.reiniciar()

    def reiniciar(self):
        self.log_algoritmo = None
        self.log_params = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SelectorModeloClasificacion":
        """
        Selecciona aleatoriamente el modelo de clasificación a usar, y configura sus
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
        