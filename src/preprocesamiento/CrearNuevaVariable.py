import numpy as np
import pandas as pd
import random

from ..RegistroTecnica import RegistroTecnica
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

class CrearNuevaVariable(BaseEstimator, TransformerMixin, RegistroTecnica):
    _instance = None  # Atributo de clase para la instancia única

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CrearNuevaVariable, cls).__new__(cls)
        return cls._instance

    def __init__(self, permitir_none=True, random_state=None):
        """
        permitir_none: si True, permite que no se cree ninguna variable nueva
        random_state: para reproducibilidad
        """
        # Evitamos re-inicializar si la instancia ya existe
        if not hasattr(self, "_initialized"):
            self.log_fase = "crear_nueva_variable"
            self.permitir_none = permitir_none
            self.random_state = random_state
            self.log_algoritmo = None
            self.log_params = None
            self._initialized = True

    def reiniciar(self):
        """
        Reinicia valores de logs de selección de técnica y parámetros para la próxima ejecución del pipeline.
        Esto es necesario porque esta clase es un singleton y se reutiliza en cada fold del pipeline
        """
        self.log_algoritmo = None
        self.log_params = None

    def _permitir_none(self, tecnicas):
        if not self.permitir_none:
            return [t for t in tecnicas if t is not None]
        return tecnicas

    def fit(self, X, y=None):
        """
        Selecciona aleatoriamente la técnica para crear nueva variable
        """
        if self.log_algoritmo is not None:
            return self
        
        generador_aleatorio = np.random.default_rng(self.random_state)
        TECNICAS = [None, "suma", "resta", "multiplicacion", "ratio", "pca"]
        TECNICAS = self._permitir_none(TECNICAS)
        self.log_algoritmo = generador_aleatorio.choice(TECNICAS)
        return self

    def transform(self, X, y=None):
        """
        Aplica la técnica seleccionada para crear una nueva variable
        """
        is_numpy = isinstance(X, np.ndarray)
        X_df = pd.DataFrame(X) if is_numpy or not isinstance(X, pd.DataFrame) else X.copy()

        if self.log_algoritmo is None:
            self.registrar_tecnica(self.log_fase, self.log_algoritmo, self.log_params)
            return X_df.values if is_numpy else X_df

        numeric_cols = [col for col in X_df.columns if pd.api.types.is_numeric_dtype(X_df[col])]

        rng = np.random.default_rng(self.random_state)
        if self.log_algoritmo in ["suma", "resta", "multiplicacion", "ratio"]:
            if len(numeric_cols) >= 2:
                # Seleccionamos dos columnas numéricas al azar
                if self.log_params is None:
                    col1, col2 = rng.choice(numeric_cols, size=2, replace=False)
                    self.log_params = (col1, col2)
                else:
                    col1, col2 = self.log_params

                if self.log_algoritmo == "suma":
                    self.registrar_tecnica(self.log_fase, self.log_algoritmo, self.log_params)
                    X_df[f'suma_{col1}_{col2}'] = X_df[col1] + X_df[col2]

                elif self.log_algoritmo == "resta":
                    self.registrar_tecnica(self.log_fase, self.log_algoritmo, self.log_params)
                    X_df[f'resta_{col1}_{col2}'] = X_df[col1] - X_df[col2]

                elif self.log_algoritmo == "multiplicacion":
                    self.registrar_tecnica(self.log_fase, self.log_algoritmo, self.log_params)
                    X_df[f'multiplicacion_{col1}_{col2}'] = X_df[col1] * X_df[col2]

                elif self.log_algoritmo == "ratio":
                    self.registrar_tecnica(self.log_fase, self.log_algoritmo, self.log_params)
                    X_df[f'ratio_{col1}_{col2}'] = X_df[col1] / (X_df[col2] + 1e-10)

        elif self.log_algoritmo == "pca":
            if len(numeric_cols) >= 2:
                if self.log_params is None:
                    n_components = rng.integers(2, len(numeric_cols))
                    self.log_params = int(n_components)
                    self.registrar_tecnica(self.log_fase, self.log_algoritmo, self.log_params)
                else:
                    n_components = self.log_params

                pca = PCA(n_components=n_components, random_state=self.random_state)
                pca_result = pca.fit_transform(X_df[numeric_cols])
                
                # Crear nuevo DataFrame SOLO con las componentes
                X_df = pd.DataFrame(
                    pca_result,
                    columns=[f"pca_{i}" for i in range(n_components)],
                    index=X_df.index
                )

        return X_df.values if is_numpy else X_df
