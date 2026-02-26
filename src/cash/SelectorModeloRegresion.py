import ast
import signal
from importlib import import_module
import pandas as pd

from src.Result import Result

from ..LLM import LLM
from ..RegistroTecnica import RegistroTecnica
from ..ExtractorMetaFeatures import ExtractorMetaFeatures

class SelectorModeloRegresion(RegistroTecnica):
    def __init__(self, config_test=None):
        super().__init__(log_fase="selector_modelo_regresion")
        self.config_test = config_test
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


    def calcular_hiper_parametros(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Selecciona aleatoriamente el modelo de regresión a usar, y configura sus
        hiperparámetros.
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
        return self
    
    @staticmethod
    def timeout_handler(signum, frame):
        raise TimeoutError("Timeout: el entrenamiento excedio el limite de tiempo")

        
    def entrenar_modelo(self, X: pd.DataFrame, y: pd.Series) -> Result[object, str]:
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

    def _get_modelo_alias(self):
        """
        Mantiene compatibilidad de nombres históricos del pipeline,
        mapeándolos a modelos disponibles en cuML.
        """
        try:
            linear_model = import_module("cuml.linear_model")
            svm = import_module("cuml.svm")
            neighbors = import_module("cuml.neighbors")
            tree = import_module("cuml.tree")
            ensemble = import_module("cuml.ensemble")
        except ImportError as error:
            raise ImportError(
                "No se pudo importar cuML. Instala RAPIDS/cuML para usar los selectores GPU."
            ) from error

        LinearRegression = linear_model.LinearRegression
        Ridge = linear_model.Ridge
        Lasso = linear_model.Lasso
        ElasticNet = linear_model.ElasticNet
        SVR = svm.SVR
        KNeighborsRegressor = neighbors.KNeighborsRegressor
        DecisionTreeRegressor = tree.DecisionTreeRegressor
        RandomForestRegressor = ensemble.RandomForestRegressor

        return {
            "lineal": lambda: LinearRegression(),
            "ridge": lambda: Ridge(),
            "lasso": lambda: Lasso(),
            "elasticnet": lambda: ElasticNet(),
            "svr": lambda: SVR(),
            "knn": lambda: KNeighborsRegressor(),
            "arbol_decision": lambda: DecisionTreeRegressor(),
            "random_forest": lambda: RandomForestRegressor(),
            "gradient_boosting": lambda: RandomForestRegressor(),
            "ada_boost": lambda: RandomForestRegressor(),
            "mlp_regressor": lambda: LinearRegression(),
        }
    
    def _get_instancia_modelo(self):
        alias_modelos = self._get_modelo_alias()
        if self.log_algoritmo not in alias_modelos:
            raise ValueError(f"Modelo no reconocido: {self.log_algoritmo}")

        return alias_modelos[self.log_algoritmo]()
        
    def _calcular_parametros(self, X: pd.DataFrame, y: pd.Series):
        llm = LLM()

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
        