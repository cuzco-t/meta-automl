import numpy as np
import pandas as pd

from ..RegistroTecnica import RegistroTecnica


class TratarDuplicados(RegistroTecnica):
    _instance = None  # Atributo de clase para almacenar la instancia única

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TratarDuplicados, cls).__new__(cls)
        return cls._instance

    def __init__(self, permitir_none=True, random_state=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        random_state: para reproducibilidad
        """
        # Evitamos re-inicializar si ya existe la instancia
        if not hasattr(self, "_initialized"):
            self.permitir_none = permitir_none
            self.random_state = random_state
            self.log_algoritmo = None
            self._initialized = True

    def reiniciar(self):
        """
        Reinicia valores de logs de selección de técnica y parámetros para la próxima ejecución del pipeline.
        Esto es necesario porque esta clase es un singleton y se reutiliza en cada fold del pipeline
        """
        self.log_algoritmo = None
        self.log_params = None

    def fit(self, X, y=None):
        """
        Decide aleatoriamente la técnica a aplicar y la guarda en self.log_algoritmo
        """
        if self.log_algoritmo is not None:
            return self

        generador_aleatorio = np.random.default_rng(self.random_state)
        TECNICAS = [None, "eliminar"]
        if not self.permitir_none:
            TECNICAS = ["eliminar"]

        self.log_algoritmo = generador_aleatorio.choice(TECNICAS)
        self.registrar_tecnica("tratar_duplicados", self.log_algoritmo, None)
        return self

    def transform(self, X, y=None):
        """
        Aplica la técnica seleccionada en fit
        """
        if self.log_algoritmo != "eliminar":
            return X if y is None else (X, y)

        # Comienza a eliminar duplicados
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        mask = ~X_df.duplicated()

        if isinstance(X, pd.DataFrame):
            X_clean = X.loc[mask]
        else:
            X_clean = X_df.loc[mask].to_numpy()

        if y is None:
            return X_clean
        else:
            if isinstance(y, pd.Series):
                y_clean = y.loc[mask]
            else:
                y_arr = np.asarray(y)
                y_clean = y_arr[mask.values]
            return X_clean, y_clean
    