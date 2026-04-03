import ast
import pandas as pd
from multiprocessing import Process, Queue

# Módulos propios (sin cambios)
from ..LLM import LLM
from ..RegistroTecnica import RegistroTecnica
from ..ExtractorMetaFeatures import ExtractorMetaFeatures
from ..Result import Result
from ..config.Configuracion import Configuracion

# Modelos con aceleración GPU (cuML)
try:
    import cuml
    from cuml.linear_model import LogisticRegression as cuLogisticRegression
    from cuml.linear_model import Ridge as cuRidge
    from cuml import SGD as cuSGD
    from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
    from cuml.ensemble import GradientBoostingClassifier as cuGradientBoostingClassifier
    from cuml import DecisionTreeClassifier as cuDecisionTreeClassifier
    from cuml.svm import SVC as cuSVC
    from cuml.svm import LinearSVC as cuLinearSVC
    from cuml.neighbors import KNeighborsClassifier as cuKNeighborsClassifier
    CUM_AVAILABLE = True
except ImportError:
    CUM_AVAILABLE = False
    # Fallback a sklearn si no hay cuML
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.neighbors import KNeighborsClassifier

# Modelos que solo existen en sklearn (sin aceleración GPU)
from sklearn.linear_model import RidgeClassifier, SGDClassifier as SkSGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier as SkGradientBoostingClassifier
from sklearn.svm import SVC as SkSVC, LinearSVC as SkLinearSVC
from sklearn.neighbors import KNeighborsClassifier as SkKNeighborsClassifier

class SelectorModeloClasificacion(RegistroTecnica):
    def __init__(self, config_test=None):
        super().__init__(log_fase="selector_modelo_clasificacion")
        self.config_test = config_test
        self.llm_seleccionado = None
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
        self.max_tiempo_entrenamiento = Configuracion().max_segundos_entrenamiento

    def calcular_hiper_parametros(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self.config_test is not None:
            self.log_algoritmo = self.config_test.get("algoritmo")
            self.log_params = self.config_test.get("params")
        else:
            self.registrar_algoritmo(self.log_algoritmo)
            self._calcular_parametros(X, y)
        self.registrar_algoritmo(self.log_algoritmo)
        return None

    def fit_model(self, modelo, X, y, queue):
        try:
            # Convertir datos a GPU (cupy o cudf) para cuML, o dejar como numpy para sklearn
            if CUM_AVAILABLE and hasattr(modelo, '_get_tags') and 'gpu' in modelo._get_tags().get('non_deterministic', ''):
                # Es un modelo cuML: convertir a cupy array
                import cupy as cp
                X_gpu = cp.asarray(X.values) if isinstance(X, pd.DataFrame) else cp.asarray(X)
                y_gpu = cp.asarray(y.values) if isinstance(y, pd.Series) else cp.asarray(y)
                modelo.fit(X_gpu, y_gpu)
            else:
                # Modelo sklearn o fallback: usar numpy
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
            return Result.fail("Timeout: el entrenamiento excedio el limite de tiempo")

        status, result = queue.get()
        if status == "ok":
            return Result.ok(result)
        else:
            return Result.fail(result)

    def _get_instancia_modelo(self):
        """Retorna la instancia del modelo (cuML si existe y está disponible, sino sklearn)."""
        if CUM_AVAILABLE:
            # Modelos con soporte nativo en cuML
            match self.log_algoritmo:
                case "logistic_regression":
                    return cuLogisticRegression()
                case "sgd_classifier":
                    # Para clasificación con SGD en cuML se usa loss='log'
                    return cuSGD(loss='log')
                case "ridge_classifier":
                    return cuRidge()
                case "decision_tree":
                    return cuDecisionTreeClassifier()
                case "random_forest":
                    return cuRandomForestClassifier()
                case "gradient_boosting":
                    return cuGradientBoostingClassifier()
                case "svc":
                    return cuSVC()
                case "linear_svc":
                    return cuLinearSVC()
                case "knn":
                    return cuKNeighborsClassifier()
                case _:
                    # Resto de algoritmos: usar sklearn (sin GPU)
                    return self._get_instancia_modelo_sklearn()
        else:
            return self._get_instancia_modelo_sklearn()

    def _get_instancia_modelo_sklearn(self):
        """Instancias de modelos sklearn (fallback o cuando no existe en cuML)."""
        match self.log_algoritmo:
            case "logistic_regression":
                return LogisticRegression()
            case "sgd_classifier":
                return SGDClassifier()
            case "ridge_classifier":
                return RidgeClassifier()
            case "decision_tree":
                return DecisionTreeClassifier()
            case "random_forest":
                return RandomForestClassifier()
            case "gradient_boosting":
                return GradientBoostingClassifier()
            case "hist_gradient_boosting":
                return HistGradientBoostingClassifier()
            case "ada_boost":
                return AdaBoostClassifier()
            case "svc":
                return SVC()
            case "linear_svc":
                return LinearSVC()
            case "knn":
                return KNeighborsClassifier()
            case "gaussian_nb":
                return GaussianNB()
            case "multinomial_nb":
                return MultinomialNB()
            case "mlp_classifier":
                return MLPClassifier()
            case "linear_discriminant_analysis":
                return LinearDiscriminantAnalysis()
            case "quadratic_discriminant_analysis":
                return QuadraticDiscriminantAnalysis()
            case _:
                raise ValueError(f"Modelo no reconocido: {self.log_algoritmo}")

    def _calcular_parametros(self, X: pd.DataFrame, y: pd.Series):
        # Sin cambios respecto al original
        llm = LLM(self.llm_seleccionado)
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