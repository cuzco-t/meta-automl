import ast
import pandas as pd
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

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
    from cuml.solvers import SGD as cuSGD
    from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
    from cuml.svm import SVC as cuSVC
    from cuml.svm import LinearSVC as cuLinearSVC
    from cuml.neighbors import KNeighborsClassifier as cuKNeighborsClassifier
    from cuml.naive_bayes import GaussianNB as cuGaussianNB
    CUM_AVAILABLE = True
except ImportError as e:
    print("cuML no disponible, se usará sklearn sin aceleración GPU")
    print(f"Error: {e}")
    CUM_AVAILABLE = False
    # Fallback a sklearn si no hay cuML
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.neighbors import KNeighborsClassifier

# Modelos que solo existen en sklearn (sin aceleración GPU)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import RidgeClassifier, SGDClassifier as SkSGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier as SkGradientBoostingClassifier
from sklearn.svm import SVC as SkSVC, LinearSVC as SkLinearSVC
from sklearn.neighbors import KNeighborsClassifier as SkKNeighborsClassifier
from sklearn.naive_bayes import GaussianNB as SkGaussianNB

from datetime import datetime

print_original = print

def print(*args, **kwargs):
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_original(f"{ahora} |", *args, **kwargs)

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

    def calcular_hiper_parametros(self, X: pd.DataFrame, y: pd.Series, meta_features) -> None:
        if self.config_test is not None:
            self.log_algoritmo = self.config_test.get("algoritmo")
            self.log_params = self.config_test.get("params")
        else:
            self.registrar_algoritmo(self.log_algoritmo)
            self._calcular_parametros(X, y, meta_features)
        self.registrar_algoritmo(self.log_algoritmo)
        return None

    def fit_model(self, modelo, es_gpu, X, y, queue):
        try:
            # Convertir datos a GPU (cupy o cudf) para cuML, o dejar como numpy para sklearn
            if es_gpu:
                # Es un modelo cuML: convertir a cupy array
                import cupy as cp
                X_gpu = cp.asarray(X.values) if isinstance(X, pd.DataFrame) else cp.asarray(X)
                y_gpu = cp.asarray(y.cat.codes.to_numpy())

                print("Entrenando modelo en GPU con cuML...")
                modelo.fit(X_gpu, y_gpu)
            else:
                print("Entrenando modelo en CPU con sklearn...")
                # Modelo sklearn o fallback: usar numpy
                modelo.fit(X.values if isinstance(X, pd.DataFrame) else X,
                           y.values if isinstance(y, pd.Series) else y)
            queue.put(("ok", modelo))
        except Exception as e:
            print(f"Error durante el entrenamiento del modelo: {e}")
            queue.put(("fail", str(e)))

    def entrenar_modelo(self, X: pd.DataFrame, y: pd.Series) -> Result[object, str]:
        modelo, es_gpu = self._get_instancia_modelo()
        hiper_parametros = self.log_params["params"]
        for param, valor in hiper_parametros.items():
            if param in modelo.get_params():
                modelo.set_params(**{param: valor})

        if "n_jobs" in modelo.get_params():
            modelo.set_params(n_jobs=-1)

        queue = mp.Queue()
        p = mp.Process(target=self.fit_model, args=(modelo, es_gpu, X, y, queue))
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
                    return cuLogisticRegression(), True
                case "sgd_classifier":
                    # Para clasificación con SGD en cuML se usa loss='log'
                    return cuSGD(loss='log'), True
                case "ridge_classifier":
                    return cuRidge(), True
                case "random_forest":
                    return cuRandomForestClassifier(), True
                case "svc":
                    return cuSVC(), True
                case "linear_svc":
                    return cuLinearSVC(), True
                case "knn":
                    return cuKNeighborsClassifier(), True
                case "gaussian_nb":
                    return cuGaussianNB(), True
                case _:
                    # Resto de algoritmos: usar sklearn (sin GPU)
                    return self._get_instancia_modelo_sklearn(), False
        else:
            return self._get_instancia_modelo_sklearn(), False

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
                return SkRandomForestClassifier()
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
                return SkGaussianNB()
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

    def _calcular_parametros(self, X: pd.DataFrame, y: pd.Series, meta_features_globales_formateadas):
        # Sin cambios respecto al original
        if self.llm_seleccionado is None:
            instancia_modelo, es_gpu = self._get_instancia_modelo()
            self.log_params["params"] = instancia_modelo.get_params()
            self.registrar_parametros(self.log_params)
            return 

        llm = LLM(self.llm_seleccionado)

        instancia_modelo, _ = self._get_instancia_modelo()

        prompt = llm.plantillas_prompts(
            plantilla="seleccionar_hiper_parametros",
            kwargs={
                "tarea": "clasificación",
                "modelo_ml": self.log_algoritmo,
                "meta_features_globales": meta_features_globales_formateadas,
                "hiper_parametros_por_defecto": instancia_modelo.get_params(),
            }
        )

        try:
            hiper_parametros_texto = llm.generar_respuesta(prompt)
            hiper_parametros = ast.literal_eval(hiper_parametros_texto)
        
        except Exception as e:
            print(f"Error al interpretar la respuesta del LLM o tiempo de espera agotado: {e}")
            print("Usando hiperparámetros por defecto.")
            hiper_parametros = instancia_modelo.get_params()

        self.log_params["params"] = hiper_parametros
        self.registrar_parametros(self.log_params)