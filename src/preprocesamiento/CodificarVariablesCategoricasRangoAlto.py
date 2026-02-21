import numpy as np
import pandas as pd

from pandas.api.types import is_object_dtype, is_categorical_dtype

from ..RegistroTecnica import RegistroTecnica

class CodificarVariablesCategoricasRangoAlto(RegistroTecnica):
    def __init__(self, permitir_none=True, semilla=None, config_test=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        semilla: para reproducibilidad
        """
        RegistroTecnica.__init__(self, log_fase="codificar_variables_categoricas_rango_alto")
        self.permitir_none = permitir_none
        self.semilla = semilla
        self.config_test = config_test
        self.ALGORITMOS = [
            "eliminar_columna"
        ]

    def _permitir_none(self, tecnicas):
        if not self.permitir_none:
            return [t for t in tecnicas if t is not None]
        return tecnicas
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Decide aleatoriamente la técnica a aplicar y la guarda en self.log_algoritmo
        """        
        if self.config_test is not None:
            self.log_algoritmo = self.config_test.get("algoritmo")
            self.log_params = self.config_test.get("params")

        else:
            self.registrar_algoritmo(self.log_algoritmo)
            self._calcular_parametros(X)

        self.registrar_algoritmo(self.log_algoritmo)
        return self
    
    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Aplica la codificación seleccionada a las variables categóricas de rango alto
        (ratio de valores únicos mayores o iguales a 0.90)
        """
        match self.log_algoritmo:
            case None:
                return X, y
            
            case "eliminar_columna":
                X_modificado = self._eliminar_columna(X)
                return X_modificado, y

            case _:
                raise ValueError(f"Técnica de codificación desconocida: {self.log_algoritmo}")
        
    def _calcular_parametros(self, X: pd.DataFrame):
        """
        Calcula y guarda en self.log_params los parámetros necesarios para la técnica seleccionada
        """
        for col in X.columns:
            if not (is_object_dtype(X[col]) or is_categorical_dtype(X[col])):
                continue

            ratio_unicos = X[col].nunique() / len(X)

            if ratio_unicos >= 0.90:
                self.log_params[col] = True
                self.registrar_parametros(self.log_params)

    def _eliminar_columna(self, X_copy: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina las columnas categóricas de rango alto codificadas en self.log_params
        
        :return: DataFrame con las columnas categóricas de rango alto eliminadas
        :rtype: DataFrame
        """
        columnas_a_eliminar_columna = list(self.log_params.keys())
        X_df = X_copy.drop(columns=columnas_a_eliminar_columna)
        return X_df