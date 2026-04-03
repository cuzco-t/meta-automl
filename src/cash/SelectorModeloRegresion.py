import ast
import pandas as pd

from src.Result import Result
from multiprocessing import Process, Queue

from ..LLM import LLM
from ..RegistroTecnica import RegistroTecnica
from ..ExtractorMetaFeatures import ExtractorMetaFeatures
from ..config.Configuracion import Configuracion

# Intentar importar cuML (GPU)
try:
    import cuml
    from cuml.linear_model import LinearRegression as cuLinearRegression
    from cuml.linear_model import Ridge as cuRidge
    from cuml.linear_model import Lasso as cuLasso
    from cuml.linear_model import ElasticNet as cuElasticNet
    from cuml.svm import SVR as cuSVR
    from cuml.neighbors import KNeighborsRegressor as cuKNeighborsRegressor
    from cuml import DecisionTreeRegressor as cuDecisionTreeRegressor
    from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
    from cuml.ensemble import GradientBoostingRegressor as cuGradientBoostingRegressor
    CUM_AVAILABLE = True
except ImportError:
    CUM_AVAILABLE = False
    # Fallback a sklearn
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.neural_network import MLPRegressor

# Modelos sin aceleración GPU (solo sklearn)
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor


class SelectorModeloRegresion(RegistroTecnica):
    def __init__(self, config_test=None):
        super().__init__(log_fase="selector_modelo_regresion")
        self.config_test = config_test
        self.llm_seleccionado = None
        self.ALGORITMOS = [
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
        self.max_tiempo_entrenamiento = Configuracion().max_segundos_entrenamiento

    def calcular_hiper_parametros(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Selecciona el modelo de regresión y configura sus hiperparámetros vía LLM.
        """
        if self.config_test is not None:
            self.log_algoritmo = self.config_test.get("algoritmo")
            self.log_params = self.config_test.get("params")
        else:
            self.registrar_algoritmo(self.log_algoritmo)
            self._calcular_parametros(X, y)

        self.registrar_algoritmo(self.log_algoritmo)
        return self  # Se mantiene el comportamiento original (aunque no se use)

    def fit_model(self, modelo, X, y, queue):
        try:
            # Determinar si el modelo es de cuML o sklearn
            if CUM_AVAILABLE and hasattr(modelo, '_get_tags') and 'gpu' in modelo._get_tags().get('non_deterministic', ''):
                # Modelo cuML: convertir a cupy
                import cupy as cp
                X_gpu = cp.asarray(X.values) if isinstance(X, pd.DataFrame) else cp.asarray(X)
                y_gpu = cp.asarray(y.values) if isinstance(y, pd.Series) else cp.asarray(y)
                modelo.fit(X_gpu, y_gpu)
            else:
                # Modelo sklearn: usar numpy
                modelo.fit(X.values if isinstance(X, pd.DataFrame) else X,
                           y.values if isinstance(y, pd.Series) else y)
            queue.put(("ok", modelo))
        except Exception as e:
            queue.put(("fail", str(e)))

    def entrenar_modelo(self, X: pd.DataFrame, y: pd.Series) -> Result[object, str]:
        modelo = self._get_instancia_modelo()
        hiper_parametros = self.log_params["params"]
        modelo.set_params(**hiper_parametros)

        queue = Queue()
        p = Process(target=self.fit_model, args=(modelo, X, y, queue))
        p.start()
        p.join(timeout=self.max_tiempo_entrenamiento)

        if p.is_alive():
            p.terminate()
            return Result.fail("Timeout: el entrenamiento excedió el límite de tiempo")

        status, result = queue.get()
        if status == "ok":
            return Result.ok(result)
        else:
            return Result.fail(result)

    def _get_instancia_modelo(self):
        """Retorna instancia del modelo (cuML si existe, sklearn si no o fallback)."""
        if CUM_AVAILABLE:
            match self.log_algoritmo:
                case "lineal":
                    return cuLinearRegression()
                case "ridge":
                    return cuRidge()
                case "lasso":
                    return cuLasso()
                case "elasticnet":
                    return cuElasticNet()
                case "svr":
                    return cuSVR()
                case "knn":
                    return cuKNeighborsRegressor()
                case "arbol_decision":
                    return cuDecisionTreeRegressor()
                case "random_forest":
                    return cuRandomForestRegressor()
                case "gradient_boosting":
                    return cuGradientBoostingRegressor()
                case "ada_boost":
                    # AdaBoost no tiene versión GPU en cuML → usar sklearn
                    return AdaBoostRegressor()
                case "mlp_regressor":
                    # MLPRegressor no está en cuML → usar sklearn
                    return MLPRegressor()
                case _:
                    raise ValueError(f"Modelo no reconocido: {self.log_algoritmo}")
        else:
            # Fallback completo a sklearn
            return self._get_instancia_modelo_sklearn()

    def _get_instancia_modelo_sklearn(self):
        """Instancias de modelos sklearn (CPU)."""
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
        """Consulta al LLM para obtener hiperparámetros del modelo seleccionado."""
        llm = LLM(self.llm_seleccionado)

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