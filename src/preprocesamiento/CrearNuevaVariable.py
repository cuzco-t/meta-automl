import numpy as np
import pandas as pd

from ..RegistroTecnica import RegistroTecnica
from sklearn.decomposition import PCA

class CrearNuevaVariable(RegistroTecnica):
    def __init__(self, permitir_none=True, semilla=None, config_test=None):
        """
        permitir_none: si True, permite que no se cree ninguna variable nueva
        semilla: para reproducibilidad
        """
        RegistroTecnica.__init__(self, log_fase="crear_nueva_variable")
        self.log_fase = "crear_nueva_variable"
        self.permitir_none = permitir_none
        self.semilla = semilla
        self.config_test = config_test
        
        self.ALGORITMOS = [
            None, 
            "llm"
        ]

    def _permitir_none(self, tecnicas):
        if not self.permitir_none:
            return [t for t in tecnicas if t is not None]
        return tecnicas

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Selecciona aleatoriamente la técnica para crear nueva variable
        """
        if self.config_test is not None:
            self.log_algoritmo = self.config_test.get("algoritmo")
            self.log_params = self.config_test.get("params")

        else:
            self.registrar_algoritmo(self.log_algoritmo)
            self._calcular_parametros(X)

        self.registrar_algoritmo(self.log_algoritmo)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Aplica la técnica seleccionada para crear una nueva variable
        """
        match self.log_algoritmo:
            case None:
                return X, y
            
            case "suma" | "resta" | "multiplicacion" | "ratio":
                X_nueva = self._crear_variable_con_operaciones_aritmeticas(X.copy())
                return X_nueva, y
            
            case "pca":
                X_nueva = self._crear_variable_con_pca(X.copy())
                return X_nueva, y
            
            case _:
                raise ValueError(f"Técnica no reconocida: {self.log_algoritmo}")

    def _calcular_parametros(self, X: pd.DataFrame):
        """
        Calcula y guarda en self.log_params los parámetros necesarios para la técnica seleccionada
        """
        cols_numericas = X.select_dtypes(include=np.number).columns
        if len(cols_numericas) < 2:
            self.registrar_parametros(self.log_params)
            return

        rng = np.random.default_rng()

        if self.log_algoritmo in ["suma", "resta", "multiplicacion", "ratio"]:
            col1, col2 = rng.choice(cols_numericas, size=2, replace=False)
            self.log_params = {"col1": col1, "col2": col2}

        elif self.log_algoritmo == "pca":
            n_components = rng.integers(2, len(cols_numericas))
            self.log_params = {"n_components": int(n_components)}

        self.registrar_parametros(self.log_params)

    def _crear_variable_con_operaciones_aritmeticas(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea una nueva variable a partir de operaciones aritméticas entre dos columnas numéricas seleccionadas al azar.
        
        :return: DataFrame con la nueva variable creada
        :rtype: DataFrame
        """
        if len(self.log_params) != 2:
            return X_df

        col1 = self.log_params["col1"]
        col2 = self.log_params["col2"]

        if self.log_algoritmo == "suma":
            X_df[f'suma_{col1}_{col2}'] = X_df[col1] + X_df[col2]

        elif self.log_algoritmo == "resta":
            X_df[f'resta_{col1}_{col2}'] = X_df[col1] - X_df[col2]

        elif self.log_algoritmo == "multiplicacion":
            X_df[f'multiplicacion_{col1}_{col2}'] = X_df[col1] * X_df[col2]

        elif self.log_algoritmo == "ratio":
            X_df[f'ratio_{col1}_{col2}'] = X_df[col1] / (X_df[col2] + 1e-10)

        return X_df
    
    def _crear_variable_con_pca(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea nuevas variables a partir de PCA aplicado a las columnas numéricas.
        
        :return: DataFrame con las nuevas variables creadas por PCA
        :rtype: DataFrame
        """
        numeric_cols = X_df.select_dtypes(include=np.number).columns
        n_components = self.log_params["n_components"]

        pca = PCA(n_components=n_components, random_state=self.semilla)
        pca_result = pca.fit_transform(X_df[numeric_cols])

        # Convertimos el resultado de PCA a DataFrame
        pca_cols = [f'PC{i+1}' for i in range(n_components)]
        df_pca = pd.DataFrame(pca_result, columns=pca_cols, index=X_df.index)

        # Solo devolvemos las columnas PCA
        return df_pca