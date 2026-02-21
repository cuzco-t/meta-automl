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
    def __init__(self, permitir_none=True, semilla=None, tarea="clasificacion", config_test=None):
        """
        permitir_none: si True, permite que no se seleccione ninguna técnica
        tarea: 'clasificacion' o 'regresion'
        semilla: para reproducibilidad
        """
        RegistroTecnica.__init__(self, log_fase="seleccionar_variables")
        self.permitir_none = permitir_none
        self.tarea = tarea
        self.semilla = semilla
        self.config_test = config_test
        self.ALGORITMOS = [
            None, 
            "variance_threshold", 
            "mutual_info",
            "select_from_model", 
            "pca_99",
            "pca_95",
            "pca_90", 
            "llm"
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
            
            case "aleatorio" | "variance_threshold" | "mutual_info" | "select_from_model" | "llm":
                X_reducido = self._seleccionar_columnas_con_parametros(X.copy())
                return X_reducido, y
            
            case "pca_99" | "pca_95" | "pca_90":
                X_pca = self._seleccionar_columnas_con_pca(X.copy())
                return X_pca, y
            
            case _:
                raise ValueError(f"Técnica de selección de variables no reconocida: {self.log_algoritmo}")                
    
    def _calcular_parametros(self, X: pd.DataFrame, y: pd.Series):
        """
        Calcula y guarda en self.log_params los parámetros necesarios para la técnica seleccionada
        """
        rng = np.random.default_rng()

        if self.log_algoritmo == "aleatorio":
            columnas_aleatorias = rng.choice(X.columns, size=rng.integers(1, X.shape[1] + 1), replace=False)
            self.log_params["columnas"] = columnas_aleatorias.tolist()
        
        elif self.log_algoritmo == "variance_threshold":
            threshold = rng.uniform(0.0, 0.2)
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(X)

            columnas_seleccionadas = X.columns[selector.get_support(indices=True)]

            self.log_params["threshold"] = threshold
            self.log_params["columnas"] = columnas_seleccionadas.tolist()

        elif self.log_algoritmo == "mutual_info":
            mi_scores = None

            if self.tarea == "clasificacion":
                mi_scores = mutual_info_classif(X, y, discrete_features=True)
            elif self.tarea == "regresion":
                mi_scores = mutual_info_regression(X, y, discrete_features=False)
            elif self.tarea == "clustering":
                return None
            
            mi_df = pd.DataFrame({'Feature': X.columns, 'MI': mi_scores})
            
            if mi_df['MI'].max() == 0:
                columnas_seleccionadas = []
            else:
                # Solo pasan aquellas con un score mayor o igual a la mitad del máximo
                threshold = 0.5 * mi_df['MI'].max()
                columnas_seleccionadas = mi_df[mi_df['MI'] >= threshold]['Feature'].tolist()

            self.log_params["threshold"] = threshold
            self.log_params["columnas"] = columnas_seleccionadas

        elif self.log_algoritmo == "select_from_model":
            model = None
            if self.tarea == "clasificacion":
                model = RandomForestClassifier(n_estimators=100, random_state=self.semilla)
            elif self.tarea == "regresion":
                model = RandomForestRegressor(n_estimators=100, random_state=self.semilla)
            elif self.tarea == "clustering":
                return None

            selector = SelectFromModel(model)
            selector.fit(X, y)

            columnas_seleccionadas = X.columns[selector.get_support(indices=True)]
            self.log_params["columnas"] = columnas_seleccionadas.tolist()

        elif self.log_algoritmo == "pca":
            cols_numericas = X.select_dtypes(include=np.number).columns
            n_components = rng.integers(2, len(cols_numericas))

            self.log_params["n_components"] = int(n_components)

        elif self.log_algoritmo == "llm":
            extractor = ExtractorMetaFeatures()
            meta_features_por_columna = extractor.extraer_meta_features_por_columna(X, y)
            meta_features_por_columna_toon = extractor.meta_features_por_columna_a_toon(meta_features_por_columna)

            llm = LLM("deepseek-r1:8b")
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

            self.log_params["columnas"] = columnas_lista

        self.registrar_parametros(self.log_params)
            
    def _seleccionar_columnas_con_parametros(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Selecciona las columnas indicadas en self.log_params["columnas"]
        """
        columnas = self.log_params.get("columnas", [])
        return X[columnas]
    
    def _seleccionar_columnas_con_pca(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica PCA con el número de componentes indicado en self.log_params["n_components"]
        """
        porcentaje_varianza = float(self.log_algoritmo.split("_")[1]) / 100
        numeric_cols = X_df.select_dtypes(include=np.number).columns

        pca = PCA(n_components=porcentaje_varianza, random_state=self.semilla)
        pca_result = pca.fit_transform(X_df[numeric_cols])

        # Convertimos el resultado de PCA a DataFrame
        pca_cols = [f'PC{i+1}' for i in range(pca_result.shape[1])]
        df_pca = pd.DataFrame(pca_result, columns=pca_cols, index=X_df.index)

        # Solo devolvemos las columnas PCA
        return df_pca
