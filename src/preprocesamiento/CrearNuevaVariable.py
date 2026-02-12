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
            self.nombre_fase = "crear_nueva_variable"
            self.permitir_none = permitir_none
            self.random_state = random_state
            self.tecnica_seleccionada_ = None
            self.parametro_tecnica_ = None
            self._initialized = True

    def reiniciar(self):
        """
        Reinicia valores de logs de selección de técnica y parámetros para la próxima ejecución del pipeline.
        Esto es necesario porque esta clase es un singleton y se reutiliza en cada fold del pipeline
        """
        self.tecnica_seleccionada_ = None
        self.parametro_tecnica_ = None

    def _permitir_none(self, tecnicas):
        if not self.permitir_none:
            return [t for t in tecnicas if t is not None]
        return tecnicas

    def fit(self, X, y=None):
        """
        Selecciona aleatoriamente la técnica para crear nueva variable
        """
        generador_aleatorio = np.random.default_rng(self.random_state)
        TECNICAS = [None, "suma", "resta", "multiplicacion", "ratio", "pca"]
        TECNICAS = self._permitir_none(TECNICAS)
        self.tecnica_seleccionada_ = generador_aleatorio.choice(TECNICAS)
        return self

    def transform(self, X, y=None):
        """
        Aplica la técnica seleccionada para crear una nueva variable
        """
        is_numpy = isinstance(X, np.ndarray)
        X_df = pd.DataFrame(X) if is_numpy or not isinstance(X, pd.DataFrame) else X.copy()

        if self.tecnica_seleccionada_ is None:
            self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)
            return X_df.values if is_numpy else X_df

        numeric_cols = [col for col in X_df.columns if pd.api.types.is_numeric_dtype(X_df[col])]

        rng = np.random.default_rng(self.random_state)
        if self.tecnica_seleccionada_ in ["suma", "resta", "multiplicacion", "ratio"]:
            if len(numeric_cols) >= 2:
                # Seleccionamos dos columnas numéricas al azar
                if self.parametro_tecnica_ is None:
                    col1, col2 = rng.choice(numeric_cols, size=2, replace=False)
                    self.parametro_tecnica_ = (col1, col2)
                else:
                    col1, col2 = self.parametro_tecnica_

                if self.tecnica_seleccionada_ == "suma":
                    X_df['suma'] = X_df[col1] + X_df[col2]
                    self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)

                elif self.tecnica_seleccionada_ == "resta":
                    X_df['resta'] = X_df[col1] - X_df[col2]
                    self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)

                elif self.tecnica_seleccionada_ == "multiplicacion":
                    X_df['multiplicacion'] = X_df[col1] * X_df[col2]
                    self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)

                elif self.tecnica_seleccionada_ == "ratio":
                    X_df['ratio'] = X_df[col1] / (X_df[col2] + 1e-10)
                    self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)

        elif self.tecnica_seleccionada_ == "pca":
            if len(numeric_cols) >= 2:
                if self.parametro_tecnica_ is None:
                    n_components = rng.integers(2, len(numeric_cols))
                    self.parametro_tecnica_ = n_components
                    self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)
                else:
                    n_components = self.parametro_tecnica_

                pca = PCA(n_components=n_components, random_state=self.random_state)
                pca_result = pca.fit_transform(X_df[numeric_cols])
                
                # Crear nuevo DataFrame SOLO con las componentes
                X_df = pd.DataFrame(
                    pca_result,
                    columns=[f"pca_{i}" for i in range(n_components)],
                    index=X_df.index
                )

        return X_df.values if is_numpy else X_df
