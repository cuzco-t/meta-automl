import ast
import toonstream
import numpy as np
import pandas as pd

from ..LLM import LLM
from ..RegistroTecnica import RegistroTecnica
from ..ExtractorMetaFeatures import ExtractorMetaFeatures
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA

class SeleccionarVariables(RegistroTecnica):
    _instance = None  # Atributo de clase para almacenar la instancia única

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SeleccionarVariables, cls).__new__(cls)
        return cls._instance

    def __init__(self, permitir_none=True, tarea='regresion', random_state=None):
        """
        permitir_none: si True, permite que no se seleccione ninguna técnica
        tarea: 'clasificacion' o 'regresion'
        random_state: para reproducibilidad
        """
        # Evitamos re-inicializar si la instancia ya existe
        if not hasattr(self, "_initialized"):
            self.nombre_fase = "seleccion_variables"
            self.permitir_none = permitir_none
            self.tarea = tarea
            self.random_state = random_state
            self.tecnica_seleccionada_ = None
            self._SEMILLA = random_state  # para modelos que requieren random_state
            self.parametro_tecnica_ = []
            self._initialized = True

    def reiniciar(self):
        self.tecnica_seleccionada_ = None
        self.parametro_tecnica_ = []

    def _permitir_none(self, tecnicas):
        if not self.permitir_none:
            return [t for t in tecnicas if t is not None]
        return tecnicas

    def fit(self, X, y=None):
        """
        Selecciona aleatoriamente la técnica de selección de variables
        """
        generador_aleatorio = np.random.default_rng(self.random_state)
        TECNICAS = [None, "aleatorio", "variance_threshold", "mutual_info",
                    "select_from_model", "pca", "llm"]
        TECNICAS.remove("llm")
        TECNICAS = self._permitir_none(TECNICAS)

        self.tecnica_seleccionada_ = generador_aleatorio.choice(TECNICAS)
        return self

    def transform(self, X, y=None):
        """
        Aplica la técnica seleccionada para reducir/eliminar variables
        """
        if y is None:
            raise ValueError("y no puede ser None para la selección de variables")
            
        is_numpy = isinstance(X, np.ndarray)
        X_df = pd.DataFrame(X) if is_numpy or not isinstance(X, pd.DataFrame) else X.copy()
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()

        n_columnas = X_df.shape[1]

        if self.tecnica_seleccionada_ is None:
            self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)
            return X_df.values if is_numpy else X_df

        rng = np.random.default_rng(self.random_state)

        if self.tecnica_seleccionada_ == "aleatorio":
            n_select = rng.integers(1, n_columnas + 1)

            if not self.parametro_tecnica_:
                columnas = rng.choice(X_df.columns, size=n_select, replace=False)
                self.parametro_tecnica_ = columnas.tolist()
                self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)

            else:
                columnas = self.parametro_tecnica_

            return X_df[columnas].values if is_numpy else X_df[columnas]

        elif self.tecnica_seleccionada_ == "variance_threshold":
            if not self.parametro_tecnica_:
                threshold = rng.uniform(0.0, 0.2)
                selector = VarianceThreshold(threshold=threshold)
                X_new = selector.fit_transform(X_df)

                columnas = X_df.columns[selector.get_support()]
                self.parametro_tecnica_ = columnas.tolist()
                self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)
            else:
                columnas = pd.Index(self.parametro_tecnica_)
                X_new = X_df[columnas]

            return pd.DataFrame(X_new, columns=columnas, index=X_df.index)

        elif self.tecnica_seleccionada_ == "mutual_info":
            if not self.parametro_tecnica_:
                n_select = rng.integers(1, n_columnas + 1)
                if self.tarea == "clasificacion":
                    mi = mutual_info_classif(X_df, y_series, random_state=self.random_state)
                elif self.tarea == "regresion":
                    mi = mutual_info_regression(X_df, y_series, random_state=self.random_state)

                mi_series = pd.Series(mi, index=X_df.columns)
                columnas = mi_series.sort_values(ascending=False).head(n_select).index
                self.parametro_tecnica_ = columnas.tolist()
                self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)
            else:
                columnas = pd.Index(self.parametro_tecnica_)

            return X_df[columnas].values if is_numpy else X_df[columnas]

        elif self.tecnica_seleccionada_ == "select_from_model":
            if not self.parametro_tecnica_:
                if self.tarea == "clasificacion":
                    model = RandomForestClassifier(n_estimators=100, random_state=self._SEMILLA)
                elif self.tarea == "regresion":
                    model = RandomForestRegressor(n_estimators=100, random_state=self._SEMILLA)

                selector = SelectFromModel(model)
                selector.fit(X_df, y_series)
                X_new = selector.transform(X_df)

                columnas = X_df.columns[selector.get_support()]
                self.parametro_tecnica_ = columnas.tolist()
                self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)
            else:
                columnas = pd.Index(self.parametro_tecnica_)
                X_new = X_df[columnas]

            return pd.DataFrame(X_new, columns=columnas, index=X_df.index)

        elif self.tecnica_seleccionada_ == "pca":
            if not self.parametro_tecnica_:
                n_components = rng.integers(2, n_columnas + 1)
                pca = PCA(n_components=n_components, random_state=self.random_state)
                X_pca = pca.fit_transform(X_df)
                columnas = [f'pca_{i}' for i in range(n_components)]
                self.parametro_tecnica_ = columnas
                self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)

            else:
                columnas = self.parametro_tecnica_
                n_components = len(columnas)
                pca = PCA(n_components=n_components, random_state=self.random_state)
                X_pca = pca.fit_transform(X_df)

            return pd.DataFrame(X_pca, columns=columnas, index=X_df.index)

        elif self.tecnica_seleccionada_ == "llm":
            extractor = ExtractorMetaFeatures()
            meta_features_por_columna = extractor.extraer_meta_features_por_columna(X, y)
            print(f"Meta-features por columna para selección de variables:\n{meta_features_por_columna}")

            llm = LLM("deepseek-r1:8b")
            meta_features_por_columna_toon = toonstream.encode(meta_features_por_columna)
            prompt = llm.plantillas_prompts(
                plantilla="seleccionar_variables",
                kwargs={
                    "tarea": self.tarea,
                    "meta_features_por_columna": meta_features_por_columna_toon
                }
            )
            input("El prompt es:\n" + prompt + "\nPresiona Enter para continuar...")
            columnas_texto = llm.generar_respuesta(prompt)
            columnas_lista = ast.literal_eval(columnas_texto)

            self.parametro_tecnica_ = columnas_lista
            self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)

        return X_df.values if is_numpy else X_df
