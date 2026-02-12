import numpy as np
import pandas as pd
from scipy import stats

from ..RegistroTecnica import RegistroTecnica
from sklearn.base import BaseEstimator, TransformerMixin

class TratarOutliersNumericos(BaseEstimator, TransformerMixin, RegistroTecnica):
    _instance = None  # Atributo de clase para la instancia única

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TratarOutliersNumericos, cls).__new__(cls)
        return cls._instance

    def __init__(self, permitir_none=True, random_state=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        random_state: para reproducibilidad
        """
        # Evitamos re-inicializar si la instancia ya existe
        if not hasattr(self, "_initialized"):
            self.nombre_fase = "tratar_outliers_numericos"
            self.permitir_none = permitir_none
            self.random_state = random_state
            self.tecnica_seleccionada_ = {}
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
        Selecciona aleatoriamente la técnica a aplicar a los outliers numéricos
        y la guarda en self.tecnica_seleccionada_
        """
        generador_aleatorio = np.random.default_rng(self.random_state)
        TECNICAS = [None, "media", "mediana", "moda", "aleatorio", "media_geometrica", "eliminar"]
        TECNICAS = self._permitir_none(TECNICAS)

        self.tecnica_seleccionada_ = generador_aleatorio.choice(TECNICAS)
        return self

    def transform(self, X, y=None):
        """
        Aplica la técnica seleccionada a los outliers numéricos
        utilizando el método IQR (1.5 * IQR)
        """
        if self.tecnica_seleccionada_ is None:
            self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, None)
            return X if y is None else (X, y)

        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        filas_a_eliminar = None

        for col in X_df.columns:
            if not pd.api.types.is_numeric_dtype(X_df[col]):
                continue

            Q1 = X_df[col].quantile(0.25)
            Q3 = X_df[col].quantile(0.75)
            IQR = Q3 - Q1

            filas_outliers = (
                (X_df[col] < (Q1 - 1.5 * IQR)) |
                (X_df[col] > (Q3 + 1.5 * IQR))
            )

            if not filas_outliers.any():
                continue

            if self.tecnica_seleccionada_ == "media":
                if self.parametro_tecnica_.get(col) is None:
                    self.parametro_tecnica_[col] = X_df[col].mean()

                X_df.loc[filas_outliers, col] = self.parametro_tecnica_[col]
                self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)

            elif self.tecnica_seleccionada_ == "mediana":
                if self.parametro_tecnica_.get(col) is None:
                    self.parametro_tecnica_[col] = X_df[col].median()

                X_df.loc[filas_outliers, col] = self.parametro_tecnica_[col]
                self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)

            elif self.tecnica_seleccionada_ == "moda":
                moda = X_df[col].mode()
                if not moda.empty:
                    if self.parametro_tecnica_.get(col) is None:
                        self.parametro_tecnica_[col] = moda.iloc[0]
                    
                    X_df.loc[filas_outliers, col] = self.parametro_tecnica_[col]
                    self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)

            elif self.tecnica_seleccionada_ == "aleatorio":
                valores_validos = X_df.loc[~filas_outliers, col].values
                if len(valores_validos) > 0:
                    X_df.loc[filas_outliers, col] = np.random.choice(
                        valores_validos, filas_outliers.sum()
                    )
                    self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, None)

            elif self.tecnica_seleccionada_ == "media_geometrica":
                valores = X_df.loc[~filas_outliers, col]
                valores_pos = valores[valores > 0]
                if not valores_pos.empty:
                    if self.parametro_tecnica_.get(col) is None:
                        self.parametro_tecnica_[col] = stats.gmean(valores_pos)

                    X_df.loc[filas_outliers, col] = self.parametro_tecnica_[col]
                    self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)

            elif self.tecnica_seleccionada_ == "eliminar":
                mask = ~filas_outliers
                filas_a_eliminar = mask if filas_a_eliminar is None else filas_a_eliminar & mask

        if self.tecnica_seleccionada_ == "eliminar" and filas_a_eliminar is not None:
            X_df = X_df.loc[filas_a_eliminar]
            self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, "outliers_eliminados")

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
