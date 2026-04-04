import ast
import numpy as np
import pandas as pd

from ..LLM import LLM
from ..RegistroTecnica import RegistroTecnica
from ..ExtractorMetaFeatures import ExtractorMetaFeatures

# Importaciones condicionales para cuML y fallback a scikit-learn
try:
    import cuml
    from cuml.decomposition import PCA as cuPCA
    from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
    from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
    from cuml.preprocessing import StandardScaler  # Aunque no se usa directamente, se deja como referencia
    CUM_AVAILABLE = True
except ImportError:
    CUM_AVAILABLE = False
    from sklearn.decomposition import PCA as skPCA
    from sklearn.ensemble import RandomForestClassifier as skRandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor as skRandomForestRegressor

# Librerías para UMAP con aceleración GPU
try:
    import cuml
    from cuml.manifold import UMAP as cuUMAP
    UMAP_GPU_AVAILABLE = True
except ImportError:
    UMAP_GPU_AVAILABLE = False
    try:
        import umap
        UMAP_CPU_AVAILABLE = True
    except ImportError:
        UMAP_CPU_AVAILABLE = False
        raise ImportError("Se requiere 'umap-learn' o 'cuml' para usar UMAP.")

# Otras importaciones de scikit-learn (se mantienen en CPU)
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA as skPCA
from sklearn.ensemble import RandomForestClassifier as skRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as skRandomForestRegressor


class SeleccionarVariables(RegistroTecnica):
    MODELOS_LLM = {
        "llm": "deepseek-r1:8b",
        "llm_deepseek-r1:8b": "deepseek-r1:8b",
        "llm_llama3.1:8b": "llama3.1:8b",
        "llm_qwen2.5-coder:7b": "qwen2.5-coder:7b",
    }

    def __init__(self, permitir_none=True, semilla=None, tarea="clasificacion", config_test=None):
        RegistroTecnica.__init__(self, log_fase="seleccionar_variables")
        self.permitir_none = permitir_none
        self.tarea = tarea
        self.semilla = semilla
        self.config_test = config_test
        self.ALGORITMOS = [
            "llm_deepseek-r1:8b",
            "llm_llama3.1:8b",
            "llm_qwen2.5-coder:7b",
            "mutual_info_25",
            "mutual_info_50",
            "mutual_info_75",
            None,
            "pca_90",
            "pca_95",
            "pca_99",
            "select_from_model",
            "umap_20",
            "umap_50",
            "umap_80",
        ]

    def _permitir_none(self, tecnicas):
        if not self.permitir_none:
            return [t for t in tecnicas if t is not None]
        return tecnicas

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Selecciona aleatoriamente la técnica de selección de variables
        """        
        if self.config_test is not None:
            self.log_algoritmo = self.config_test.get("algoritmo")
            self.log_params = self.config_test.get("params")

        else:
            self.registrar_algoritmo(self.log_algoritmo)
            self._calcular_parametros(X, y)

        self.registrar_algoritmo(self.log_algoritmo)
        return self

    def transform(self, X, y=None):
        """
        Aplica la técnica seleccionada para reducir/eliminar variables
        """
        match self.log_algoritmo:
            case None:
                return X, y
            
            case "aleatorio" | "select_from_model" | "llm" | "llm_deepseek-r1:8b" | "llm_llama3.1:8b" | "llm_qwen2.5-coder:7b":
                X_reducido = self._seleccionar_columnas_con_parametros(X.copy())
                return X_reducido, y
            
            case "mutual_info_25" | "mutual_info_50" | "mutual_info_75":
                X_reducido = self._seleccionar_columnas_con_parametros(X.copy())
                return X_reducido, y
            
            case "pca_99" | "pca_95" | "pca_90":
                X_pca = self._seleccionar_columnas_con_pca(X.copy())
                return X_pca, y
            
            case "umap_20" | "umap_50" | "umap_80":
                X_umap = self.seleccionar_columnas_con_umap(X.copy())
                return X_umap, y
            
            case _:
                raise ValueError(f"Técnica de selección de variables no reconocida: {self.log_algoritmo}")                
    
    def _calcular_parametros(self, X: pd.DataFrame, y: pd.Series):
        """
        Calcula y guarda en self.log_params los parámetros necesarios para la técnica seleccionada.
        Versión optimizada para GPU cuando esté disponible.
        """
        if self.log_algoritmo is None:
            self.log_params = {}
            self.registrar_parametros(self.log_params)
            return

        rng = np.random.default_rng(42)

        if self.log_algoritmo == "aleatorio":
            columnas_aleatorias = rng.choice(X.columns, size=rng.integers(1, X.shape[1] + 1), replace=False)
            self.log_params["columnas"] = columnas_aleatorias.tolist()

        elif "mutual_info" in self.log_algoritmo:
            # La información mutua se mantiene en CPU por ahora
            mi_scores = None
            if self.tarea == "clasificacion":
                mi_scores = mutual_info_classif(X, y, discrete_features=True)
            elif self.tarea == "regresion":
                mi_scores = mutual_info_regression(X, y, discrete_features=False)
            elif self.tarea == "clustering":
                return None

            mi_df = pd.DataFrame({'Feature': X.columns, 'MI': mi_scores})

            if mi_df['MI'].max() == 0:
                threshold = 0
                columnas_seleccionadas = []
            else:
                threshold = (float(self.log_algoritmo.split("_")[-1]) / 100) * mi_df['MI'].max()
                columnas_seleccionadas = mi_df[mi_df['MI'] >= threshold]['Feature'].tolist()

            self.log_params["threshold"] = threshold
            self.log_params["columnas"] = columnas_seleccionadas

        elif self.log_algoritmo == "select_from_model":
            # Selección basada en Random Forest, ahora con aceleración GPU si está disponible
            if CUM_AVAILABLE:
                if self.tarea == "clasificacion":
                    model = cuRandomForestClassifier(n_estimators=100, random_state=42)
                elif self.tarea == "regresion":
                    model = cuRandomForestRegressor(n_estimators=100, random_state=42)
                elif self.tarea == "clustering":
                    return None

                # Convertir a GPU para entrenar el modelo
                import cupy as cp
                X_gpu = cp.asarray(X.values)
                y_gpu = cp.asarray(y.values)
                model.fit(X_gpu, y_gpu)

                # Obtener importancias de características
                importancias = model.feature_importances_
                # Seleccionar características con importancia > 0 (o un umbral)
                umbral = importancias.mean()
                columnas_seleccionadas = X.columns[importancias > umbral].tolist()
            else:
                # Fallback a CPU
                if self.tarea == "clasificacion":
                    model = skRandomForestClassifier(n_estimators=100, random_state=42)
                elif self.tarea == "regresion":
                    model = skRandomForestRegressor(n_estimators=100, random_state=42)
                elif self.tarea == "clustering":
                    return None

                selector = SelectFromModel(model)
                selector.fit(X, y)
                columnas_seleccionadas = X.columns[selector.get_support(indices=True)].tolist()

            self.log_params["columnas"] = columnas_seleccionadas

        elif self.log_algoritmo in ["pca_90", "pca_95", "pca_99"]:
            self.log_params["porcentaje_varianza"] = float(self.log_algoritmo.split("_")[1]) / 100

        elif self.log_algoritmo in ["umap_20", "umap_50", "umap_80"]:
            porcentaje_componentes = int(self.log_algoritmo.split("_")[1]) / 100
            columnas_numericas = X.select_dtypes(include=np.number).columns
            cantidad_columnas_numericas = len(columnas_numericas)

            n_components = max(
                2,
                min(int(cantidad_columnas_numericas * porcentaje_componentes), cantidad_columnas_numericas - 1)
            )
            self.log_params["n_components"] = int(n_components)

        elif self.log_algoritmo in self.MODELOS_LLM:
            # Sin cambios: la selección por LLM se mantiene igual
            extractor = ExtractorMetaFeatures()
            meta_features_por_columna = extractor.extraer_meta_features_por_columna(X, y)
            meta_features_por_columna_toon = extractor.formatear_meta_features_por_columna(meta_features_por_columna)

            llm = LLM(self.MODELOS_LLM[self.log_algoritmo])
            prompt = llm.plantillas_prompts(
                plantilla="seleccionar_variables",
                kwargs={
                    "tarea": self.tarea,
                    "meta_features_por_columna": meta_features_por_columna_toon
                }
            )

            try:
                columnas_texto = llm.generar_respuesta(prompt)
                columnas_lista = ast.literal_eval(columnas_texto)

            except Exception as e:
                print(f"Error al interpretar la respuesta del LLM o tiempo de espera agotado: {e}")
                raise ValueError(f"Error al interpretar la respuesta del LLM: {e}")


            columnas_validas = [columna for columna in columnas_lista if columna in X.columns]
            self.log_params["columnas"] = columnas_validas

        self.registrar_parametros(self.log_params)
            
    def _seleccionar_columnas_con_parametros(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Selecciona las columnas indicadas en self.log_params["columnas"]
        """
        columnas = self.log_params.get("columnas", [])
        return X[columnas]
    
    def _seleccionar_columnas_con_pca(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Aplica PCA con el porcentaje de varianza indicado, usando GPU si está disponible."""
        porcentaje_varianza = self.log_params["porcentaje_varianza"]
        numeric_cols = X_df.select_dtypes(include=np.number).columns

        if False:
            # Versión GPU con cuML
            import cupy as cp
            X_numeric_gpu = cp.asarray(X_df[numeric_cols].values)
            pca = cuPCA(n_components=porcentaje_varianza)
            pca_result_gpu = pca.fit_transform(X_numeric_gpu)
            pca_result = cp.asnumpy(pca_result_gpu)  # Convertir a numpy para compatibilidad
        else:
            # Fallback a CPU con scikit-learn
            pca = skPCA(n_components=porcentaje_varianza, random_state=self.semilla)
            pca_result = pca.fit_transform(X_df[numeric_cols])

        pca_cols = [f'PC{i+1}' for i in range(pca_result.shape[1])]
        df_pca = pd.DataFrame(pca_result, columns=pca_cols, index=X_df.index)
        return df_pca

    def seleccionar_columnas_con_umap(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Aplica UMAP con el número de componentes indicado, usando GPU si está disponible."""
        n_components = self.log_params["n_components"]
        numeric_cols = X_df.select_dtypes(include=np.number).columns

        if UMAP_GPU_AVAILABLE:
            # Versión GPU con cuML
            import cupy as cp
            X_numeric_gpu = cp.asarray(X_df[numeric_cols].values)
            reducer = cuUMAP(n_components=n_components, random_state=42)
            embedding_gpu = reducer.fit_transform(X_numeric_gpu)
            embedding = cp.asnumpy(embedding_gpu)
        elif UMAP_CPU_AVAILABLE:
            # Fallback a CPU con umap-learn
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            embedding = reducer.fit_transform(X_df[numeric_cols])
        else:
            raise ImportError("No se encontró ninguna implementación de UMAP. Instala 'umap-learn' o 'cuml'.")

        umap_cols = [f'UMAP{i+1}' for i in range(embedding.shape[1])]
        df_umap = pd.DataFrame(embedding, columns=umap_cols, index=X_df.index)
        return df_umap
