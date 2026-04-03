import ast
import pandas as pd
from multiprocessing import Process, Queue

from ..LLM import LLM
from ..RegistroTecnica import RegistroTecnica
from ..ExtractorMetaFeatures import ExtractorMetaFeatures
from ..Result import Result
from ..config.Configuracion import Configuracion

# Intentar importar cuML (GPU)
try:
    import cuml
    from cuml.cluster import KMeans as cuKMeans
    from cuml.cluster import DBSCAN as cuDBSCAN
    from cuml import AgglomerativeClustering as cuAgglomerativeClustering
    from cuml.cluster import SpectralClustering as cuSpectralClustering
    CUM_AVAILABLE = True
except ImportError as e:
    print(f"Error al importar cuML: {e}")
    CUM_AVAILABLE = False
    # Fallback a sklearn
    from sklearn.cluster import KMeans, DBSCAN

# Algoritmos sin aceleración GPU (solo sklearn)
from sklearn.cluster import MeanShift, Birch
from sklearn.cluster import KMeans, DBSCAN  # para fallback explícito

from datetime import datetime

print_original = print

def print(*args, **kwargs):
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_original(f"{ahora} |", *args, **kwargs)

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
        self.max_tiempo_entrenamiento = Configuracion().max_segundos_entrenamiento

    def calcular_hiper_parametros(self, X: pd.DataFrame):
        """
        Selecciona el modelo de clustering y configura sus hiperparámetros vía LLM.
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
            # Determinar si el modelo es de cuML o sklearn
            if CUM_AVAILABLE and hasattr(modelo, '_get_tags') and 'gpu' in modelo._get_tags().get('non_deterministic', ''):
                # Modelo cuML: convertir X a cupy
                import cupy as cp
                X_gpu = cp.asarray(X.values) if isinstance(X, pd.DataFrame) else cp.asarray(X)

                print("Entrenando modelo en GPU con cuML...")
                modelo.fit(X_gpu)
            else:
                # Modelo sklearn: usar numpy
                print("Entrenando modelo en CPU con sklearn...")
                modelo.fit(X.values if isinstance(X, pd.DataFrame) else X)
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
        """Retorna instancia del modelo (cuML si existe, sklearn si no)."""
        if CUM_AVAILABLE:
            match self.log_algoritmo:
                case "kmeans":
                    return cuKMeans()
                case "dbscan":
                    return cuDBSCAN()
                case "agglomerative_clustering":
                    return cuAgglomerativeClustering()
                case "spectral_clustering":
                    return cuSpectralClustering()
                case _:
                    # Resto de algoritmos sin soporte GPU → usar sklearn
                    return self._get_instancia_modelo_sklearn()
        else:
            return self._get_instancia_modelo_sklearn()

    def _get_instancia_modelo_sklearn(self):
        """Instancias de modelos sklearn (CPU)."""
        match self.log_algoritmo:
            case "kmeans":
                return KMeans()
            case "dbscan":
                return DBSCAN()
            case "agglomerative_clustering":
                return AgglomerativeClustering()
            case "mean_shift":
                return MeanShift()
            case "spectral_clustering":
                return SpectralClustering()
            case "birch":
                return Birch()
            case _:
                raise ValueError(f"Modelo no reconocido: {self.log_algoritmo}")

    def _calcular_parametros(self, X: pd.DataFrame):
        """Consulta al LLM para obtener hiperparámetros del modelo seleccionado."""
        if self.llm_seleccionado is None:
            self.log_params["params"] = self._get_instancia_modelo().get_params()
            self.registrar_parametros(self.log_params)
            return
            
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