import numpy as np
import pandas as pd
from scipy import stats

from ..RegistroTecnica import RegistroTecnica

class TratarOutliersNumericos(RegistroTecnica):
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
            self.log_fase = "tratar_outliers_numericos"
            self.permitir_none = permitir_none
            self.random_state = random_state
            self.log_algoritmo = {}
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
        Selecciona aleatoriamente la técnica a aplicar a los outliers numéricos
        y la guarda en self.log_algoritmo
        """
        if self.log_algoritmo is not None:
            return self
            
        generador_aleatorio = np.random.default_rng(self.random_state)
        TECNICAS = [None, "media", "mediana", "moda", "aleatorio", "media_geometrica", "eliminar"]
        TECNICAS = self._permitir_none(TECNICAS)

        self.log_algoritmo = generador_aleatorio.choice(TECNICAS)
        return self

    def transform(self, X, y=None):
        """
        Aplica la técnica seleccionada a los outliers numéricos
        utilizando el método IQR (1.5 * IQR)
        """
        if self.log_algoritmo is None:
            self.registrar_tecnica(self.log_fase, self.log_algoritmo, self.log_params)
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

            if self.log_algoritmo == "media":
                if self.log_params.get(col) is None:
                    self.log_params[col] = X_df[col].mean()

                self.registrar_tecnica(self.log_fase, self.log_algoritmo, self.log_params)
                X_df.loc[filas_outliers, col] = self.log_params[col]

            elif self.log_algoritmo == "mediana":
                if self.log_params.get(col) is None:
                    self.log_params[col] = X_df[col].median()

                self.registrar_tecnica(self.log_fase, self.log_algoritmo, self.log_params)
                X_df.loc[filas_outliers, col] = self.log_params[col]

            elif self.log_algoritmo == "moda":
                moda = X_df[col].mode()
                if not moda.empty:
                    if self.log_params.get(col) is None:
                        self.log_params[col] = moda.iloc[0]
                    
                    self.registrar_tecnica(self.log_fase, self.log_algoritmo, self.log_params)
                    X_df.loc[filas_outliers, col] = self.log_params[col]

            elif self.log_algoritmo == "aleatorio":
                valores_validos = X_df.loc[~filas_outliers, col].values
                if len(valores_validos) > 0:
                    self.registrar_tecnica(self.log_fase, self.log_algoritmo, None)
                    X_df.loc[filas_outliers, col] = np.random.choice(
                        valores_validos, filas_outliers.sum()
                    )

            elif self.log_algoritmo == "media_geometrica":
                valores = X_df.loc[~filas_outliers, col]
                valores_pos = valores[valores > 0]
                if not valores_pos.empty:
                    if self.log_params.get(col) is None:
                        self.log_params[col] = stats.gmean(valores_pos)

                    self.registrar_tecnica(self.log_fase, self.log_algoritmo, self.log_params)
                    X_df.loc[filas_outliers, col] = self.log_params[col]

            elif self.log_algoritmo == "eliminar":
                mask = ~filas_outliers
                filas_a_eliminar = mask if filas_a_eliminar is None else filas_a_eliminar & mask

        if self.log_algoritmo == "eliminar" and filas_a_eliminar is not None:
            self.registrar_tecnica(self.log_fase, self.log_algoritmo, "outliers_eliminados")
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
