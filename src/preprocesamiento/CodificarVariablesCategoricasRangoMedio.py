import numpy as np
import pandas as pd

from pandas.api.types import is_object_dtype, is_categorical_dtype

from ..RegistroTecnica import RegistroTecnica

class CodificarVariablesCategoricasRangoMedio(RegistroTecnica):
    def __init__(self, permitir_none=True, random_state=None, config_test=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        random_state: para reproducibilidad
        """
        RegistroTecnica.__init__(self, log_fase="codificar_variables_categoricas_rango_medio")
        self.permitir_none = permitir_none
        self.random_state = random_state
        self.config_test = config_test
        self.ALGORITMOS = [
            None, 
            "frequency_encoding", 
            "eliminar_variable"
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
        Aplica la codificación seleccionada a las variables categóricas de rango medio
        (ratio de valores únicos entre 0.05 y 0.90)
        """
        match self.log_algoritmo:
            case None:
                return X, y
            
            case "frequency_encoding":
                X_codificado = self._codificar_frequency_encoding(X.copy())
                return X_codificado, y
            
            case "eliminar_variable":
                X_codificado = self._eliminar_variable(X.copy())
                return X_codificado, y
            
            case _:
                raise ValueError(f"Técnica de codificación desconocida: {self.log_algoritmo}")
        
    def _calcular_parametros(self, X: pd.DataFrame) -> None:
        """
        Calcula y guarda en self.log_params los parámetros necesarios para la técnica seleccionada
        """
        for col in X.columns:
            if not (is_object_dtype(X[col]) or is_categorical_dtype(X[col])):
                continue

            ratio_unicos = X[col].nunique() / len(X)
            if 0.05 < ratio_unicos < 0.90:
                if self.log_algoritmo == "frequency_encoding":
                    self.log_params[col] = X[col].value_counts(normalize=True)

                elif self.log_algoritmo == "eliminar_variable":
                    self.log_params[col] = True

            self.registrar_parametros(self.log_params)

    def _codificar_frequency_encoding(self, X_copy: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica frequency encoding a las columnas categóricas de rango medio codificadas en self.log_params
        
        :return: DataFrame con las columnas categóricas de rango medio codificadas
        :rtype: DataFrame
        """
        for col, freqs in self.log_params.items():
            if col in X_copy.columns:
                frecuencia_minima = freqs.min()
                X_copy[col] = X_copy[col].map(freqs).fillna(frecuencia_minima).astype(float)
                print(f"Columna '{col}': {X_copy[col].isna().sum()}")

        return X_copy
    
    def _eliminar_variable(self, X_copy: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina las columnas categóricas de rango medio codificadas en self.log_params
        
        :return: DataFrame con las columnas categóricas de rango medio eliminadas
        :rtype: DataFrame
        """
        columnas_a_eliminar = list(self.log_params.keys())
        return X_copy.drop(columns=columnas_a_eliminar)
