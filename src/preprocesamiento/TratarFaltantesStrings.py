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
            self.log_fase = "tratar_faltantes_strings"
            self.log_algoritmo = None
            self.permitir_none = permitir_none
            self.random_state = random_state
            self.log_params = {}
            self._initialized = True

    def reiniciar(self):
        """
        Reinicia valores de logs de selección de técnica y parámetros para la próxima ejecución del pipeline.
        Esto es necesario porque esta clase es un singleton y se reutiliza en cada fold del pipeline
        """
        self.log_algoritmo = None
        self.log_params = {}

    def _permitir_none(self, tecnicas):
        if not self.permitir_none:
            return [t for t in tecnicas if t is not None]
        return tecnicas
    

    def fit(self, X, y=None):
        """
        Selecciona aleatoriamente la técnica a aplicar a los valores faltantes
        de tipo string y la guarda en self.log_algoritmo
        """
        if self.log_algoritmo is not None:
            return self
        
        generador_aleatorio = np.random.default_rng(self.random_state)
        TECNICAS = [None, "moda", "aleatorio", "eliminar", "etiqueta_desconocido"]
        TECNICAS = self._permitir_none(TECNICAS)

        self.log_algoritmo = generador_aleatorio.choice(TECNICAS)
        return self

    def transform(self, X, y=None):
        """
        Aplica la técnica seleccionada a los valores faltantes de tipo string
        """
        if self.log_algoritmo is None:
            self.registrar_tecnica(self.log_fase, self.log_algoritmo, self.log_params)
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

            if self.log_algoritmo == "moda":
                moda = X_df[col].mode()
                if not moda.empty:
                    if self.log_params.get(col) is None:
                        self.log_params[col] = moda.iloc[0]

                    self.registrar_tecnica(self.log_fase, self.log_algoritmo, self.log_params)
                    X_df[col] = X_df[col].fillna(self.log_params[col])

            elif self.log_algoritmo == "aleatorio":
                valores_validos = X_df[col].dropna().values
                if len(valores_validos) > 0:
                    self.registrar_tecnica(self.log_fase, self.log_algoritmo, "valores_validos")
                    X_df.loc[X_df[col].isna(), col] = np.random.choice(
                        valores_validos, X_df[col].isna().sum()
                    )

            elif self.log_algoritmo == "etiqueta_desconocido":
                if self.log_params.get(col) is None:
                    self.log_params[col] = "DESCONOCIDO"
                    
                self.registrar_tecnica(self.log_fase, self.log_algoritmo, self.log_params)
                X_df[col] = X_df[col].fillna(self.log_params[col])

            elif self.log_algoritmo == "eliminar":
                mask = X_df[col].notna()
                filas_a_eliminar = mask if filas_a_eliminar is None else filas_a_eliminar & mask

        if self.log_algoritmo == "eliminar" and filas_a_eliminar is not None:
            self.registrar_tecnica(self.log_fase, self.log_algoritmo, "filas_nulas")
            X_df = X_df.loc[filas_a_eliminar]

        if y is None:
            return X_df
        else:
            if self.log_algoritmo == "eliminar" and filas_a_eliminar is not None:
                if isinstance(y, pd.Series):
                    y_clean = y.loc[filas_a_eliminar]
                else:
                    y_arr = np.asarray(y)
                    y_clean = y_arr[filas_a_eliminar.values]
                return X_df, y_clean
            return X_df, y
