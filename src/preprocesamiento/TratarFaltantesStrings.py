import numpy as np
import pandas as pd

from ..RegistroTecnica import RegistroTecnica
from sklearn.base import BaseEstimator, TransformerMixin

class TratarFaltantesStrings(BaseEstimator, TransformerMixin, RegistroTecnica):
    _instance = None  # Atributo de clase para almacenar la instancia única

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TratarFaltantesStrings, cls).__new__(cls)
        return cls._instance

    def __init__(self, permitir_none=True, random_state=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        random_state: para reproducibilidad
        """
        # Evitamos re-inicializar si ya existe la instancia
        if not hasattr(self, "_initialized"):
            self.nombre_fase = "tratar_faltantes_strings"
            self.tecnica_seleccionada_ = None
            self.permitir_none = permitir_none
            self.random_state = random_state
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
        Selecciona aleatoriamente la técnica a aplicar a los valores faltantes
        de tipo string y la guarda en self.tecnica_seleccionada_
        """
        if self.tecnica_seleccionada_ is not None:
            return self
        
        generador_aleatorio = np.random.default_rng(self.random_state)
        TECNICAS = [None, "moda", "aleatorio", "eliminar", "etiqueta_desconocido"]
        TECNICAS = self._permitir_none(TECNICAS)

        self.tecnica_seleccionada_ = generador_aleatorio.choice(TECNICAS)
        return self

    def transform(self, X, y=None):
        """
        Aplica la técnica seleccionada a los valores faltantes de tipo string
        """
        if self.tecnica_seleccionada_ is None:
            self.registrar_tecnica(self.nombre_fase, None, None)
            return X if y is None else (X, y)

        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        filas_a_eliminar = None

        for col in X_df.columns:
            es_texto = (
                X_df[col].dtype == "object"
                or pd.api.types.is_string_dtype(X_df[col])
            )

            if not (es_texto and X_df[col].isna().any()):
                continue

            if self.tecnica_seleccionada_ == "moda":
                moda = X_df[col].mode()
                if not moda.empty:
                    if self.parametro_tecnica_.get(col) is None:
                        self.parametro_tecnica_[col] = moda.iloc[0]

                    X_df[col] = X_df[col].fillna(self.parametro_tecnica_[col])
                    self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)

            elif self.tecnica_seleccionada_ == "aleatorio":
                valores_validos = X_df[col].dropna().values
                if len(valores_validos) > 0:
                    X_df.loc[X_df[col].isna(), col] = np.random.choice(
                        valores_validos, X_df[col].isna().sum()
                    )
                    self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, "valores_validos")

            elif self.tecnica_seleccionada_ == "etiqueta_desconocido":
                if self.parametro_tecnica_.get(col) is None:
                    self.parametro_tecnica_[col] = "DESCONOCIDO"
                    
                X_df[col] = X_df[col].fillna(self.parametro_tecnica_[col])
                self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)

            elif self.tecnica_seleccionada_ == "eliminar":
                mask = X_df[col].notna()
                filas_a_eliminar = mask if filas_a_eliminar is None else filas_a_eliminar & mask

        if self.tecnica_seleccionada_ == "eliminar" and filas_a_eliminar is not None:
            X_df = X_df.loc[filas_a_eliminar]
            self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, "filas_nulas")

        if y is None:
            return X_df
        else:
            if self.tecnica_seleccionada_ == "eliminar" and filas_a_eliminar is not None:
                if isinstance(y, pd.Series):
                    y_clean = y.loc[filas_a_eliminar]
                else:
                    y_arr = np.asarray(y)
                    y_clean = y_arr[filas_a_eliminar.values]
                return X_df, y_clean
            return X_df, y
