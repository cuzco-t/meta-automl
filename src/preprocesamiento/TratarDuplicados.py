import numpy as np
import pandas as pd

from ..RegistroTecnica import RegistroTecnica


class TratarDuplicados(RegistroTecnica):
    _instance = None  # Atributo de clase para almacenar la instancia única

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TratarDuplicados, cls).__new__(cls)
        return cls._instance

    def __init__(self, permitir_none=True, semilla=None, config_test=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        semilla: para reproducibilidad
        """
        # Evitamos re-inicializar si ya existe la instancia
        if not hasattr(self, "_initialized"):
            RegistroTecnica.__init__(self, log_fase="tratar_duplicados")
            self.permitir_none = permitir_none
            self.semilla = semilla
            self.config_test = config_test
            self.reiniciar()
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
        Decide aleatoriamente la técnica a aplicar y la guarda en self.log_algoritmo
        """
        if self.log_algoritmo is not None:
            return self
        
        if self.config_test is not None:
            self.log_algoritmo = self.config_test.get("algoritmo")
            self.log_params = self.config_test.get("params")

        else:
            generador_aleatorio = np.random.default_rng()
            TECNICAS = self._permitir_none([None, "eliminar"])

            self.log_algoritmo = generador_aleatorio.choice(TECNICAS)

        self.registrar_tecnica("tratar_duplicados", self.log_algoritmo, None)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Aplica la técnica seleccionada en fit
        :return: X e y con la técnica aplicada
        :rtype: tuple[pd.DataFrame, pd.Series]
        """

        match self.log_algoritmo:
            case None:
                return X, y
            case "eliminar":
                return self._eliminar(X, y)
            case _:
                raise ValueError(f"Técnica desconocida: {self.log_algoritmo}")
    
    def _eliminar(self, X_df: pd.DataFrame, y_df: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Elimina filas duplicadas en X_df y las correspondientes en y_df
        
        :return: X_df y y_df sin filas duplicadas
        :rtype: tuple[pd.DataFrame, pd.Series]
        """
        mask = ~X_df.duplicated()

        X_clean = X_df.loc[mask]
        y_clean = y_df.loc[mask]

        return X_clean, y_clean