import numpy as np
import pandas as pd
from scipy import stats

from ..RegistroTecnica import RegistroTecnica
from sklearn.base import BaseEstimator, TransformerMixin

class NormalizarDatosNumericos(BaseEstimator, TransformerMixin, RegistroTecnica):
    _instance = None  # Atributo de clase para la instancia única

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(NormalizarDatosNumericos, cls).__new__(cls)
        return cls._instance

    def __init__(self, permitir_none=True, random_state=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        random_state: para reproducibilidad
        """
        # Evitamos re-inicializar si la instancia ya existe
        if not hasattr(self, "_initialized"):
            self.nombre_fase = "normalizar_datos_numericos"
            self.permitir_none = permitir_none
            self.random_state = random_state
            self.tecnica_seleccionada_ = None
            self.parametro_tecnica_ = {}
            self._initialized = True

    def reiniciar(self):
        """
        Reinicia valores de logs de selección de técnica y parámetros para la próxima ejecución del pipeline.
        Esto es necesario porque esta clase es un singleton y se reutiliza en cada fold del pipeline
        """
        self.tecnica_seleccionada_ = None
        self.parametro_tecnica_ = {}

    def _permitir_none(self, tecnicas):
        if not self.permitir_none:
            return [t for t in tecnicas if t is not None]
        return tecnicas

    def fit(self, X, y=None):
        """
        Selecciona aleatoriamente la técnica de normalización
        """
        generador_aleatorio = np.random.default_rng(self.random_state)
        TECNICAS = [None, "z-score", "box-cox", "cuadrado", "sqrt", "ln", "inverso"]
        TECNICAS = self._permitir_none(TECNICAS)

        self.tecnica_seleccionada_ = generador_aleatorio.choice(TECNICAS)
        return self

    def transform(self, X, y=None):
        """
        Aplica la técnica seleccionada a las columnas numéricas
        """
        is_numpy = isinstance(X, np.ndarray)
        X_df = pd.DataFrame(X) if is_numpy or not isinstance(X, pd.DataFrame) else X.copy()

        if self.tecnica_seleccionada_ is None:
            return X_df.values if is_numpy else X_df

        for col in X_df.columns:
            if not pd.api.types.is_numeric_dtype(X_df[col]):
                continue

            if self.tecnica_seleccionada_ == "z-score":
                if self.parametro_tecnica_.get(col) is None:
                    self.parametro_tecnica_[col] = {
                        "mean": X_df[col].mean(),
                        "std": X_df[col].std(ddof=0)
                    }
                    
                X_df[col] = (X_df[col] - self.parametro_tecnica_[col]["mean"]) / self.parametro_tecnica_[col]["std"]

            elif self.tecnica_seleccionada_ == "box-cox":
                # Solo si todos los valores son positivos
                if (X_df[col] > 0).all():
                    X_df[col], _ = stats.boxcox(X_df[col])

            elif self.tecnica_seleccionada_ == "cuadrado":
                X_df[col] = X_df[col] ** 2

            elif self.tecnica_seleccionada_ == "sqrt":
                X_df[col] = np.sqrt(np.abs(X_df[col]))

            elif self.tecnica_seleccionada_ == "ln":
                X_df[col] = np.log1p(np.abs(X_df[col]))

            elif self.tecnica_seleccionada_ == "inverso":
                X_df[col] = 1 / (1 + np.abs(X_df[col]))

        self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)

        return X_df.values if is_numpy else X_df
