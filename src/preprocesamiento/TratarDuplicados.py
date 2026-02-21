import numpy as np
import pandas as pd

from ..RegistroTecnica import RegistroTecnica


class TratarDuplicados(RegistroTecnica):
    def __init__(self, permitir_none=True, semilla=None, config_test=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        semilla: para reproducibilidad
        """
        RegistroTecnica.__init__(self, log_fase="tratar_duplicados")
        self.permitir_none = permitir_none
        self.semilla = semilla
        self.config_test = config_test
        self.ALGORITMOS = [
            "eliminar",
            None
        ]

    def _permitir_none(self, tecnicas):
        if not self.permitir_none:
            return [t for t in tecnicas if t is not None]
        return tecnicas

    def fit(self, X, y=None):
        """
        Decide aleatoriamente la técnica a aplicar y la guarda en self.log_algoritmo
        """
        if self.config_test is not None:
            self.log_algoritmo = self.config_test.get("algoritmo")
            self.log_params = self.config_test.get("params")

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
                return self._eliminar(X, y if y is not None else None)
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
        y_clean = y_df.loc[mask] if y_df is not None else None

        return X_clean, y_clean