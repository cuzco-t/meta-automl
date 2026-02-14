import numpy as np
import pandas as pd

from ..RegistroTecnica import RegistroTecnica
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

class EscalarDatosNumericos(BaseEstimator, TransformerMixin, RegistroTecnica):
    _instance = None  # Atributo de clase para almacenar la instancia única

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EscalarDatosNumericos, cls).__new__(cls)
        return cls._instance

    def __init__(self, permitir_none=True, random_state=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        random_state: para reproducibilidad
        """
        # Evitamos re-inicializar si la instancia ya existe
        if not hasattr(self, "_initialized"):
            self.nombre_fase = "escalar_datos_numericos"
            self.permitir_none = permitir_none
            self.random_state = random_state
            self.tecnica_seleccionada_ = None
            self.scaler_ = None  # guardaremos el objeto scaler para transform
            self._initialized = True

    def reiniciar(self):
        """
        Reinicia valores de logs de selección de técnica y parámetros para la próxima ejecución del pipeline.
        Esto es necesario porque esta clase es un singleton y se reutiliza en cada fold del pipeline
        """
        self.tecnica_seleccionada_ = None
        self.parametro_tecnica_ = {}
        self.scaler_ = None

    def _permitir_none(self, tecnicas):
        if not self.permitir_none:
            return [t for t in tecnicas if t is not None]
        return tecnicas

    def fit(self, X, y=None):
        """
        Selecciona aleatoriamente la técnica de escalado y prepara el scaler si es necesario
        """
        if self.tecnica_seleccionada_ is not None:
            return self
        
        generador_aleatorio = np.random.default_rng(self.random_state)
        TECNICAS = [None, "min-max", "max-abs-scaler", "standard-scaler", "robust-scaler"]
        TECNICAS = self._permitir_none(TECNICAS)

        self.tecnica_seleccionada_ = generador_aleatorio.choice(TECNICAS)
        self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, None)

        # Solo inicializamos el scaler si se seleccionó alguna técnica
        if self.tecnica_seleccionada_ == "min-max":
            self.scaler_ = MinMaxScaler()
        elif self.tecnica_seleccionada_ == "standard-scaler":
            self.scaler_ = StandardScaler()
        elif self.tecnica_seleccionada_ == "robust-scaler":
            self.scaler_ = RobustScaler()
        elif self.tecnica_seleccionada_ == "max-abs-scaler":
            self.scaler_ = MaxAbsScaler()
        else:
            self.scaler_ = None

        # Ajustamos solo columnas numéricas si hay scaler
        if self.scaler_ is not None:
            X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
            cols_numericas = X_df.select_dtypes(include=np.number).columns
            if len(cols_numericas) > 0:
                self.scaler_.fit(X_df[cols_numericas])

        return self

    def transform(self, X, y=None):
        """
        Aplica el escalado seleccionado a las columnas numéricas
        """
        is_numpy = isinstance(X, np.ndarray)
        X_df = pd.DataFrame(X) if is_numpy or not isinstance(X, pd.DataFrame) else X.copy()

        if self.tecnica_seleccionada_ is None:
            return X if is_numpy else X_df

        cols_numericas = X_df.select_dtypes(include=np.number).columns

        if self.scaler_ is not None and len(cols_numericas) > 0:
            X_scaled = self.scaler_.transform(X_df[cols_numericas])
            X_df[cols_numericas] = X_scaled

        return X_df.values if is_numpy else X_df
